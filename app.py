import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

# ==========================================
# 1. KONFIGURASI HALAMAN & ANTARMUKA (UI)
# ==========================================
st.set_page_config(
    page_title="ACS Meteorological Dashboard - WMO/ICAO Standard",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        .bmkg-header {
            background: linear-gradient(135deg, #0B3C5D 0%, #1D2731 50%, #328CC1 100%);
            padding: 22px; border-radius: 12px; color: white; margin-bottom: 25px;
            box-shadow: 0 6px 12px rgba(0,0,0,0.15); border-left: 6px solid #FBC02D;
        }
        .bmkg-header h1 { margin: 0; font-size: 26px; font-weight: 700; letter-spacing: 0.5px; }
        .bmkg-header p { margin: 6px 0 0 0; font-size: 14px; opacity: 0.92; font-weight: 300; }
        div[data-testid="stMetricValue"] { color: #0B3C5D; font-size: 24px; font-weight: bold; }
        .stAlert { border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# Palet Warna MEJIKUHIBINIU Standar
PALET_MEJIKUHIBINIU = [
    "#D32F2F", "#F57C00", "#FBC02D", "#388E3C", 
    "#1976D2", "#303F9F", "#7B1FA2", "#C2185B", 
    "#0097A7", "#455A64"
]

MONTHS_ID = [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni", 
    "Juli", "Agustus", "September", "Oktober", "November", "Desember"
]

# ==========================================
# 2. SMART DATA PARSERS
# ==========================================
def parse_synoptic_adaptive(df, is_temp=False):
    valid_rows = []
    format_type = "synoptic_yr_dt"

    for idx, row in df.iterrows():
        try:
            val0_str = str(row.iloc[0]).strip().replace(',', '.')
            val1_str = str(row.iloc[1]).strip().replace(',', '.')
            if not (val0_str.replace('.', '', 1).isdigit() and val1_str.replace('.', '', 1).isdigit()):
                continue
            
            v0, v1 = float(val0_str), float(val1_str)
            
            if 2000 <= v0 <= 2100 and 1 <= v1 <= 31:
                valid_rows.append([int(v0), int(v1)] + list(row.iloc[2:].values))
                format_type = "synoptic_yr_dt"
            elif 1 <= v0 <= 31 and 2000 <= v1 <= 2100:
                valid_rows.append([int(v1), int(v0)] + list(row.iloc[2:].values))
                format_type = "synoptic_yr_dt"
            elif 0 <= v0 <= 23 and 2000 <= v1 <= 2100:
                valid_rows.append([int(v1), int(v0)] + list(row.iloc[2:].values))
                format_type = "hourly_row"
            elif 2000 <= v0 <= 2100 and 0 <= v1 <= 23:
                valid_rows.append([int(v0), int(v1)] + list(row.iloc[2:].values))
                format_type = "hourly_row"
        except Exception:
            continue
            
    if not valid_rows: return pd.DataFrame()
        
    parsed_df = pd.DataFrame(valid_rows)
    
    if format_type == "synoptic_yr_dt":
        num_cols = len(parsed_df.columns) - 2
        if num_cols >= 24:
            hours_cols = [f"{h:02d}" for h in range(24)]
            parsed_df = parsed_df.iloc[:, :2+len(hours_cols)]
            parsed_df.columns = ['Tahun', 'Tanggal'] + hours_cols
        elif num_cols >= 8:
            base_hours = ['00', '03', '06', '09', '12', '15', '18', '21']
            if not is_temp and num_cols >= 11:
                cols = ['Tahun', 'Tanggal'] + base_hours + ['Mean', 'Max', 'Min']
            else:
                cols = ['Tahun', 'Tanggal'] + base_hours
            parsed_df = parsed_df.iloc[:, :len(cols)]
            parsed_df.columns = cols
        else:
            hours_cols = [f"{h*3:02d}" for h in range(num_cols)]
            parsed_df.columns = ['Tahun', 'Tanggal'] + hours_cols
            
    elif format_type == "hourly_row":
        if len(parsed_df.columns) >= 3:
            parsed_df = parsed_df.iloc[:, :3]
            parsed_df.columns = ['Tahun', 'Jam', 'Nilai']
        else: return pd.DataFrame()
            
    for c in parsed_df.columns:
        parsed_df[c] = pd.to_numeric(parsed_df[c], errors='coerce').fillna(0)
    return parsed_df

def parse_hourly_freq(df, col_names):
    valid_data = []
    for idx, row in df.iterrows():
        try:
            val0_str = str(row.iloc[0]).strip().replace(',', '.')
            val1_str = str(row.iloc[1]).strip().replace(',', '.')
            if not (val0_str.replace('.', '', 1).isdigit() and val1_str.replace('.', '', 1).isdigit()): continue
            v0, v1 = float(val0_str), float(val1_str)
            
            if 0 <= v0 <= 23 and 2000 <= v1 <= 2100:
                valid_data.append([int(v0), int(v1)] + list(row.iloc[2:].values))
            elif 2000 <= v0 <= 2100 and 0 <= v1 <= 23:
                valid_data.append([int(v1), int(v0)] + list(row.iloc[2:].values))
        except Exception: continue
            
    if not valid_data: return pd.DataFrame()
    parsed_df = pd.DataFrame(valid_data).iloc[:, :len(col_names)]
    parsed_df.columns = col_names[:len(parsed_df.columns)]
    for c in parsed_df.columns:
        parsed_df[c] = pd.to_numeric(parsed_df[c], errors='coerce').fillna(0)
    return parsed_df

def parse_wind(df):
    valid_data = []
    current_year = 2021
    wind_sectors = ['35-36-01', '02-03-04', '05-06-07', '08-09-10', '11-12-13', '14-15-16', 
                    '17-18-19', '20-21-22', '23-24-25', '26-27-28', '29-30-31', '32-33-34']
    
    for idx, row in df.iterrows():
        try:
            val0 = str(row.iloc[0]).strip()
            val1 = str(row.iloc[1]).strip().replace(' ', '').upper()
            val2_str = str(row.iloc[2]).strip().replace(' ', '').upper() if len(row) > 2 else ""
            
            if val0.isdigit() and len(val0) == 4 and 2000 <= int(val0) <= 2100: current_year = int(val0)
                
            target_dir, start_col_idx = None, 2
            if val1 in ['CALM', 'VARIABLE'] or val1 in wind_sectors: target_dir, start_col_idx = val1, 2
            elif val2_str in ['CALM', 'VARIABLE'] or val2_str in wind_sectors: target_dir, start_col_idx = val2_str, 3
                
            if target_dir:
                yr = int(val0) if (val0.isdigit() and len(val0) == 4 and 2000 <= int(val0) <= 2100) else current_year
                valid_data.append([yr, target_dir] + list(row.iloc[start_col_idx:].values))
        except Exception: continue
            
    if not valid_data: return pd.DataFrame()
    parsed_df = pd.DataFrame(valid_data)
    expected_cols = ['Tahun', 'Direction', '1-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-45', '>45', 'Total']
    parsed_df = parsed_df.iloc[:, :len(expected_cols)]
    parsed_df.columns = expected_cols[:len(parsed_df.columns)]
    for c in parsed_df.columns[2:]: parsed_df[c] = pd.to_numeric(parsed_df[c], errors='coerce').fillna(0)
    return parsed_df

# ==========================================
# 3. ENGINE PEMUAT DATA
# ==========================================
@st.cache_data(show_spinner=False)
def load_all_data():
    datasets = {k: pd.DataFrame() for k in ['Temp', 'TempMaxMin', 'RH', 'HS', 'Vis', 'Wind']}
    
    def get_file_path(filename):
        if os.path.exists(filename): return filename
        if os.path.exists(os.path.join('data', filename)): return os.path.join('data', filename)
        for f in os.listdir('.'):
            if f.lower() == filename.lower(): return f
        return filename

    files = {
        'Temp': (get_file_path('Temp_2021-2025.xlsx'), lambda df: parse_synoptic_adaptive(df, is_temp=True)),
        'TempMaxMin': (get_file_path('TempMaxMin_2021-2025.xlsx'), lambda df: parse_synoptic_adaptive(df, is_temp=False)),
        'RH': (get_file_path('RH_2021-2025.xlsx'), lambda df: parse_synoptic_adaptive(df, is_temp=False)),
        'HS': (get_file_path('HS_2021-2025.xlsx'), lambda df: parse_hourly_freq(df, ['Jam', 'Tahun', '<150', '<200', '<300', '<500', '<1000', '<1500'])),
        'Vis': (get_file_path('Vis_2021-2025.xlsx'), lambda df: parse_hourly_freq(df, ['Jam', 'Tahun', '<200', '<400', '<600', '<800', '<1500', '<1800', '<3000', '<5000', '<8000'])),
        'Wind': (get_file_path('Wind_2021-2025.xlsx'), parse_wind)
    }
    
    for key, (filepath, parser) in files.items():
        if os.path.exists(filepath):
            all_sheets = []
            try:
                xls = pd.ExcelFile(filepath, engine='openpyxl')
                for month_idx, month_name in enumerate(MONTHS_ID):
                    matched_sheet = None
                    for s_name in xls.sheet_names:
                        s_clean, m_clean = str(s_name).strip().lower(), month_name.lower()
                        if (s_clean == m_clean or s_clean.startswith(m_clean) or 
                            s_clean.startswith(m_clean[:3]) or f"{month_idx+1:02d}" in s_clean):
                            matched_sheet = s_name
                            break
                    if matched_sheet:
                        df_parsed = parser(pd.read_excel(xls, sheet_name=matched_sheet, header=None))
                        if not df_parsed.empty:
                            df_parsed['Bulan'] = month_name
                            df_parsed['Bulan_Idx'] = month_idx + 1
                            all_sheets.append(df_parsed)
                if all_sheets: datasets[key] = pd.concat(all_sheets, ignore_index=True)
            except Exception as e: st.sidebar.error(f"⚠️ Gagal memproses {filepath}: {str(e)}")
    return datasets

with st.spinner("🔄 Mengekstrak Laporan Climatological WMO..."):
    data = load_all_data()

# ==========================================
# 4. NAVIGASI SIDEBAR & FILTER
# ==========================================
st.sidebar.markdown("### 🧭 Navigasi & Filter ACS")
st.sidebar.markdown("---")

param_options = {
    "Temp": "1. Suhu Udara Synoptic (°C)",
    "TempMaxMin": "2. Suhu Ekstrem Max/Min (°C)",
    "RH": "3. Kelembapan Relatif / RH (%)",
    "Visibility": "4. Jarak Pandang / Visibility (m)",
    "Cloud Base (HS)": "5. Tinggi Dasar Awan / Ceiling (ft)",
    "Wind": "6. Mawar Angin / Wind Rose (Knots)"
}

selected_param = st.sidebar.selectbox("Pilih Parameter Meteorologi:", list(param_options.keys()), format_func=lambda x: param_options[x])
month_choice = st.sidebar.selectbox("Filter Bulan (Khusus Menu 1):", ["Semua Bulan"] + MONTHS_ID)
selected_year = st.sidebar.selectbox("Filter Tahun:", ["Semua Tahun"] + [2021, 2022, 2023, 2024, 2025])

st.sidebar.markdown("---")
st.sidebar.caption("💡 **Standar ICAO/WMO:** Seluruh observasi waktu menggunakan standar UTC Synoptic. Arah angin mengindikasikan arah datangnya angin.")

def filter_df(df, ignore_month=False):
    if df.empty: return df
    res = df.copy()
    if not ignore_month and month_choice != "Semua Bulan": res = res[res['Bulan'] == month_choice]
    if selected_year != "Semua Tahun": res = res[res['Tahun'] == int(selected_year)]
    return res

# ==========================================
# 5. HEADER & METRIK UTAMA (KPI)
# ==========================================
st.markdown("""
    <div class="bmkg-header">
        <h1>Dashboard Climatological Summary (ACS) & Diurnal Analysis</h1>
        <p>Sistem Pemantauan Terintegrasi Parameter Meteorologi Permukaan Standar WMO/ICAO</p>
    </div>
""", unsafe_allow_html=True)

kpi_cols = st.columns(4)
with kpi_cols[0]:
    t_df = filter_df(data['TempMaxMin'])
    st.metric("Rerata Suhu Udara", f"{t_df['Mean'].mean():.1f} °C" if (not t_df.empty and 'Mean' in t_df.columns) else "N/A")
with kpi_cols[1]:
    r_df = filter_df(data['RH'])
    st.metric("Rerata Kelembapan (RH)", f"{r_df['Mean'].mean():.1f} %" if (not r_df.empty and 'Mean' in r_df.columns) else "N/A")
with kpi_cols[2]:
    w_df = filter_df(data['Wind'])
    st.metric("Frekuensi Angin CALM", f"{w_df[w_df['Direction'] == 'CALM']['Total'].mean():.1f} %" if not w_df.empty else "N/A")
with kpi_cols[3]:
    v_df = filter_df(data['Vis'])
    st.metric("Visibilitas < 8000m", f"{v_df['<8000'].mean():.1f} %" if (not v_df.empty and '<8000' in v_df.columns) else "N/A")

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 6. ENGINE VISUALISASI (2 TABS UTAMA)
# ==========================================
def apply_wmo_style(fig, title_text, x_label, y_label):
    fig.update_layout(
        title=dict(text=f"<b>{title_text}</b>", font=dict(size=16, color="#0B3C5D")),
        xaxis_title=f"<b>{x_label}</b>", yaxis_title=f"<b>{y_label}</b>",
        margin=dict(l=20, r=20, t=50, b=50), template="plotly_white", hovermode="x unified",
        legend=dict(title="<b>Kategori Parameter:</b>", orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
    return fig

# Helper fungsi agregasi standar tunggal (Temp, RH)
def aggregate_single_param(df, time_col='Jam_UTC'):
    if 'Jam' in df.columns and 'Nilai' in df.columns:
        agg = df.groupby('Jam')['Nilai'].mean().reset_index().rename(columns={'Jam': time_col, 'Nilai': 'Value'})
        if time_col == 'Jam_UTC': agg[time_col] = agg[time_col].apply(lambda x: f"{int(x):02d}")
    else:
        syn_hours = [f"{h:02d}" for h in range(24)]
        avail = [h for h in syn_hours if h in df.columns] or [col for col in df.columns if str(col).replace('.', '', 1).isdigit() and 0 <= float(col) <= 23]
        if not avail: return pd.DataFrame()
        melted = df.melt(id_vars=['Tahun', 'Bulan_Idx', 'Tanggal', 'Bulan'], value_vars=avail, var_name=time_col, value_name='Value')
        agg = melted.groupby([time_col])['Value'].mean().reset_index()
        if time_col == 'Jam_UTC': agg[time_col] = agg[time_col].apply(lambda x: f"{int(float(str(x))):02d}")
    return agg.sort_values(time_col)

# Helper fungsi agregasi musiman tunggal (Temp, RH)
def aggregate_seasonal_single(df):
    if df.empty: return pd.DataFrame()
    if 'Jam' in df.columns and 'Nilai' in df.columns:
        return df.groupby(['Bulan_Idx', 'Bulan'])['Nilai'].mean().reset_index().rename(columns={'Nilai': 'Value'}).sort_values('Bulan_Idx')
    else:
        syn_hours = [f"{h:02d}" for h in range(24)]
        avail = [h for h in syn_hours if h in df.columns] or [col for col in df.columns if str(col).replace('.', '', 1).isdigit() and 0 <= float(col) <= 23]
        melted = df.melt(id_vars=['Tahun', 'Bulan_Idx', 'Bulan'], value_vars=avail, var_name='Jam_UTC', value_name='Value')
        return melted.groupby(['Bulan_Idx', 'Bulan'])['Value'].mean().reset_index().sort_values('Bulan_Idx')

tab1, tab2 = st.tabs(["📊 Menu 1: Pola Diurnal (Berdampingan)", "🌦️ Menu 2: Pola Musiman (Tren Bulanan)"])

# ----------------- TAB 1: DIURNAL (BERDAMPINGAN) -----------------
with tab1:
    col_met, col_hist = st.columns(2)
    
    if selected_param in ["Temp", "TempMaxMin", "RH"]:
        df_filtered = filter_df(data[selected_param])
        if df_filtered.empty: st.warning(f"⚠️ Data {selected_param} kosong/tidak valid.")
        else:
            agg_df = aggregate_single_param(df_filtered)
            y_label = "Suhu (°C)" if "Temp" in selected_param else "Kelembapan (%)"
            param_name = param_options[selected_param].split(". ")[1]
            agg_df['Parameter'] = param_name # Force legend creation
            
            # Meteogram
            fig_line = px.line(agg_df, x='Jam_UTC', y='Value', color='Parameter', markers=True, color_discrete_sequence=[PALET_MEJIKUHIBINIU[0] if "Temp" in selected_param else PALET_MEJIKUHIBINIU[4]])
            fig_line.update_traces(line=dict(width=3.5), marker=dict(size=9))
            fig_line = apply_wmo_style(fig_line, f"Meteogram Diurnal - {month_choice} ({selected_year})", "Jam (UTC)", y_label)
            col_met.plotly_chart(fig_line, use_container_width=True)
            
            # Histogram/Bar
            fig_bar = px.bar(agg_df, x='Jam_UTC', y='Value', color='Parameter', barmode='group', color_discrete_sequence=[PALET_MEJIKUHIBINIU[1] if "Temp" in selected_param else PALET_MEJIKUHIBINIU[3]])
            fig_bar = apply_wmo_style(fig_bar, f"Histogram Diurnal - {month_choice} ({selected_year})", "Jam (UTC)", y_label)
            col_hist.plotly_chart(fig_bar, use_container_width=True)

    elif selected_param in ["Visibility", "Cloud Base (HS)"]:
        df_filtered = filter_df(data['Vis'] if selected_param == "Visibility" else data['HS'])
        if df_filtered.empty: st.warning(f"⚠️ Data {selected_param} kosong/tidak valid.")
        else:
            cols = ['<200', '<400', '<600', '<800', '<1500', '<1800', '<3000', '<5000', '<8000'] if selected_param == "Visibility" else ['<150', '<200', '<300', '<500', '<1000', '<1500']
            avail_cols = [c for c in cols if c in df_filtered.columns]
            agg_v = df_filtered.groupby('Jam')[avail_cols].mean().reset_index().sort_values('Jam')
            hm_df = agg_v.melt(id_vars='Jam', value_vars=avail_cols, var_name='Kategori Parameter', value_name='Frekuensi (%)')
            
            # Meteogram
            fig_line = px.line(hm_df, x='Jam', y='Frekuensi (%)', color='Kategori Parameter', markers=True, color_discrete_sequence=PALET_MEJIKUHIBINIU)
            fig_line.update_traces(line=dict(width=2.8), marker=dict(size=6))
            fig_line = apply_wmo_style(fig_line, f"Meteogram {selected_param} - {month_choice}", "Jam (UTC)", "Frekuensi (%)")
            fig_line.update_layout(xaxis=dict(tickmode='linear', dtick=3))
            col_met.plotly_chart(fig_line, use_container_width=True)
            
            # Histogram/Bar
            fig_bar = px.bar(hm_df, x='Jam', y='Frekuensi (%)', color='Kategori Parameter', barmode='group', color_discrete_sequence=PALET_MEJIKUHIBINIU)
            fig_bar = apply_wmo_style(fig_bar, f"Histogram {selected_param} - {month_choice}", "Jam (UTC)", "Frekuensi (%)")
            fig_bar.update_layout(xaxis=dict(tickmode='linear', dtick=3))
            col_hist.plotly_chart(fig_bar, use_container_width=True)

    elif selected_param == "Wind":
        df_w = filter_df(data['Wind'])
        if df_w.empty: st.warning("⚠️ Data Wind kosong/tidak valid.")
        else:
            dir_map = {'35-36-01':'N', '02-03-04':'NNE', '05-06-07':'ENE', '08-09-10':'E', '11-12-13':'ESE', '14-15-16':'SSE', '17-18-19':'S', '20-21-22':'SSW', '23-24-25':'WSW', '26-27-28':'W', '29-30-31':'WNW', '32-33-34':'NNW'}
            rose_df = df_w[~df_w['Direction'].isin(['CALM', 'VARIABLE'])].copy()
            if not rose_df.empty:
                rose_df['Arah'] = rose_df['Direction'].map(dir_map)
                avail_speeds = [s for s in ['1-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-45', '>45'] if s in rose_df.columns]
                melt_rose = rose_df.melt(id_vars=['Arah'], value_vars=avail_speeds, var_name='Speed (Knot)', value_name='Freq')
                agg_rose = melt_rose.groupby(['Arah', 'Speed (Knot)'])['Freq'].mean().reset_index()
                
                # Polar Rose
                fig_polar = px.bar_polar(agg_rose, r="Freq", theta="Arah", color="Speed (Knot)", color_discrete_sequence=PALET_MEJIKUHIBINIU, template="plotly_white", title=f"<b>Wind Rose Pola Polar</b>")
                fig_polar.update_layout(legend=dict(orientation="h", y=-0.2))
                col_met.plotly_chart(fig_polar, use_container_width=True)
                
                # Histogram Distribusi Arah
                fig_bar = px.bar(agg_rose, x='Arah', y='Freq', color='Speed (Knot)', barmode='stack', color_discrete_sequence=PALET_MEJIKUHIBINIU)
                fig_bar = apply_wmo_style(fig_bar, "Histogram Distribusi Angin", "Arah Mata Angin", "Frekuensi (%)")
                col_hist.plotly_chart(fig_bar, use_container_width=True)

# ----------------- TAB 2: MUSIMAN (SEASONAL) -----------------
with tab2:
    st.info(f"💡 Menampilkan pola **TREN MUSIMAN** dari Januari - Desember berdasarkan dataset murni (tanpa modifikasi). Filter Bulan di sidebar diabaikan untuk menu ini. (Tahun: {selected_year})")
    
    if selected_param in ["Temp", "TempMaxMin", "RH"]:
        df_season = filter_df(data[selected_param], ignore_month=True)
        if not df_season.empty:
            agg_season = aggregate_seasonal_single(df_season)
            if not agg_season.empty:
                y_label = "Suhu (°C)" if "Temp" in selected_param else "Kelembapan (%)"
                param_name = param_options[selected_param].split(". ")[1]
                agg_season['Parameter'] = param_name # Force legend
                
                fig_season = px.line(agg_season, x='Bulan', y='Value', color='Parameter', markers=True, color_discrete_sequence=[PALET_MEJIKUHIBINIU[3]])
                fig_season.update_traces(line=dict(width=4), marker=dict(size=10))
                fig_season = apply_wmo_style(fig_season, f"Meteogram Pola Musiman ({param_name}) - {selected_year}", "Bulan Observasi", y_label)
                st.plotly_chart(fig_season, use_container_width=True)

    elif selected_param in ["Visibility", "Cloud Base (HS)"]:
        df_season = filter_df(data['Vis'] if selected_param == "Visibility" else data['HS'], ignore_month=True)
        if not df_season.empty:
            cols = ['<200', '<400', '<600', '<800', '<1500', '<1800', '<3000', '<5000', '<8000'] if selected_param == "Visibility" else ['<150', '<200', '<300', '<500', '<1000', '<1500']
            avail_cols = [c for c in cols if c in df_season.columns]
            agg_v_season = df_season.groupby(['Bulan_Idx', 'Bulan'])[avail_cols].mean().reset_index().sort_values('Bulan_Idx')
            hm_df_s = agg_v_season.melt(id_vars='Bulan', value_vars=avail_cols, var_name='Kategori Parameter', value_name='Frekuensi (%)')
            
            fig_season = px.line(hm_df_s, x='Bulan', y='Frekuensi (%)', color='Kategori Parameter', markers=True, color_discrete_sequence=PALET_MEJIKUHIBINIU)
            fig_season.update_traces(line=dict(width=3), marker=dict(size=8))
            fig_season = apply_wmo_style(fig_season, f"Meteogram Pola Musiman {selected_param} - {selected_year}", "Bulan Observasi", "Frekuensi (%)")
            st.plotly_chart(fig_season, use_container_width=True)

    elif selected_param == "Wind":
        df_season = filter_df(data['Wind'], ignore_month=True)
        if not df_season.empty:
            calm_df = df_season[df_season['Direction'] == 'CALM'].copy()
            if not calm_df.empty:
                agg_calm = calm_df.groupby(['Bulan_Idx', 'Bulan'])['Total'].mean().reset_index().sort_values('Bulan_Idx')
                agg_calm['Parameter'] = "Angin CALM"
                
                fig_season = px.line(agg_calm, x='Bulan', y='Total', color='Parameter', markers=True, color_discrete_sequence=[PALET_MEJIKUHIBINIU[5]])
                fig_season.update_traces(line=dict(width=4), marker=dict(size=10))
                fig_season = apply_wmo_style(fig_season, f"Tren Musiman Kejadian Angin CALM - {selected_year}", "Bulan Observasi", "Frekuensi Kejadian (%)")
                st.plotly_chart(fig_season, use_container_width=True)
            else: st.info("Tidak ada data angin CALM musiman tercatat untuk ditampilkan.")
