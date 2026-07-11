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

PALET_KATEGORI = px.colors.qualitative.Plotly

MONTHS_ID = [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni", 
    "Juli", "Agustus", "September", "Oktober", "November", "Desember"
]

# ==========================================
# 2. SMART DATA PARSERS 
# ==========================================
def parse_synoptic(df):
    valid_rows = []
    for idx, row in df.iterrows():
        try:
            val0 = str(row.iloc[0]).strip().replace(',', '.')
            val1 = str(row.iloc[1]).strip().replace(',', '.')
            if not (val0.replace('.', '', 1).isdigit() and val1.replace('.', '', 1).isdigit()): continue
            v0, v1 = float(val0), float(val1)
            
            if 2000 <= v0 <= 2100 and 1 <= v1 <= 31:
                valid_rows.append([int(v0), int(v1)] + list(row.iloc[2:].values))
            elif 1 <= v0 <= 31 and 2000 <= v1 <= 2100:
                valid_rows.append([int(v1), int(v0)] + list(row.iloc[2:].values))
        except Exception: continue
            
    if not valid_rows: return pd.DataFrame()
        
    parsed_df = pd.DataFrame(valid_rows)
    num_cols = len(parsed_df.columns)
    
    base_cols = ['Tahun', 'Tanggal', '00', '03', '06', '09', '12', '15', '18', '21', 'Daily_Mean', 'Max', 'Min']
    
    if num_cols >= 13:
        parsed_df = parsed_df.iloc[:, :13]
        parsed_df.columns = base_cols
    elif num_cols >= 10:
        parsed_df = parsed_df.iloc[:, :10]
        parsed_df.columns = base_cols[:10]
    else:
        return pd.DataFrame()
            
    for c in parsed_df.columns:
        # PERBAIKAN FATAL: Menghapus .fillna(0) agar data tetap otentik dan akurat.
        # Nilai kosong/tidak ada (NaN) tidak boleh direpresentasikan sebagai 0 dlm standar WMO.
        parsed_df[c] = pd.to_numeric(parsed_df[c], errors='coerce')
    return parsed_df

def parse_hourly_freq(df, col_names):
    valid_data = []
    for idx, row in df.iterrows():
        try:
            val0 = str(row.iloc[0]).strip().replace(',', '.')
            val1 = str(row.iloc[1]).strip().replace(',', '.')
            if not (val0.replace('.', '', 1).isdigit() and val1.replace('.', '', 1).isdigit()): continue
            v0, v1 = float(val0), float(val1)
            
            if 0 <= v0 <= 23 and 2000 <= v1 <= 2100:
                valid_data.append([int(v0), int(v1)] + list(row.iloc[2:].values))
            elif 2000 <= v0 <= 2100 and 0 <= v1 <= 23:
                valid_data.append([int(v1), int(v0)] + list(row.iloc[2:].values))
        except Exception: continue
            
    if not valid_data: return pd.DataFrame()
    parsed_df = pd.DataFrame(valid_data).iloc[:, :len(col_names)]
    parsed_df.columns = col_names[:len(parsed_df.columns)]
    for c in parsed_df.columns:
        # PERBAIKAN FATAL
        parsed_df[c] = pd.to_numeric(parsed_df[c], errors='coerce')
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
                
                vals = [str(x).replace(',', '.') for x in row.iloc[start_col_idx:].values]
                valid_data.append([yr, target_dir] + vals)
        except Exception: continue
            
    if not valid_data: return pd.DataFrame()
    parsed_df = pd.DataFrame(valid_data)
    expected_cols = ['Tahun', 'Direction', '1-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-45', '>45', 'Total']
    parsed_df = parsed_df.iloc[:, :len(expected_cols)]
    parsed_df.columns = expected_cols[:len(parsed_df.columns)]
    for c in parsed_df.columns[2:]: 
        # PERBAIKAN FATAL
        parsed_df[c] = pd.to_numeric(parsed_df[c], errors='coerce')
    return parsed_df

# ==========================================
# 3. ENGINE PEMUAT DATA
# ==========================================
@st.cache_data(show_spinner=False)
def load_all_data():
    datasets = {k: pd.DataFrame() for k in ['TempMaxMin', 'TempFreq', 'RH', 'HS', 'Vis', 'Wind']}
    
    def get_file_path(filename):
        if os.path.exists(filename): return filename
        if os.path.exists(os.path.join('data', filename)): return os.path.join('data', filename)
        return filename

    files = {
        'TempMaxMin': (get_file_path('TempMaxMin_2021-2025.xlsx'), parse_synoptic),
        'TempFreq': (get_file_path('Temp_2021-2025.xlsx'), lambda df: parse_hourly_freq(df, ['Jam', 'Tahun', '5 - 0', '0 - 5', '5 - 10', '10 - 15', '15 - 20', '20 - 25', '25 - 30', '30 - 35', '> 35'])),
        'RH': (get_file_path('RH_2021-2025.xlsx'), parse_synoptic),
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
    "TempMaxMin": "1. Suhu Udara Synoptic (°C)",
    "TempFreq": "2. Frekuensi Distribusi Suhu (%)",
    "RH": "3. Kelembapan Relatif / RH (%)",
    "Vis": "4. Jarak Pandang / Visibility (%)",
    "HS": "5. Tinggi Dasar Awan / Ceiling (%)",
    "Wind": "6. Mawar Angin / Wind Rose (%)"
}

selected_param = st.sidebar.selectbox("Pilih Parameter Meteorologi:", list(param_options.keys()), format_func=lambda x: param_options[x])
month_choice = st.sidebar.selectbox("Filter Bulan:", ["Semua Bulan"] + MONTHS_ID)
selected_year = st.sidebar.selectbox("Filter Tahun:", ["Semua Tahun"] + [2021, 2022, 2023, 2024, 2025])

st.sidebar.markdown("---")
st.sidebar.caption("💡 **Standar ICAO/WMO:** Seluruh observasi waktu menggunakan standar UTC Synoptic.")

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
    st.metric("Rerata Suhu Udara", f"{t_df['Daily_Mean'].mean():.1f} °C" if (not t_df.empty and 'Daily_Mean' in t_df.columns) else "N/A")
with kpi_cols[1]:
    r_df = filter_df(data['RH'])
    st.metric("Rerata Kelembapan (RH)", f"{r_df['Daily_Mean'].mean():.1f} %" if (not r_df.empty and 'Daily_Mean' in r_df.columns) else "N/A")
with kpi_cols[2]:
    w_df = filter_df(data['Wind'])
    st.metric("Frekuensi Angin CALM", f"{w_df[w_df['Direction'] == 'CALM']['Total'].mean():.1f} %" if not w_df.empty else "N/A")
with kpi_cols[3]:
    v_df = filter_df(data['Vis'])
    st.metric("Visibilitas < 8000m", f"{v_df['<8000'].mean():.1f} %" if (not v_df.empty and '<8000' in v_df.columns) else "N/A")

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 6. ENGINE VISUALISASI TUNGGAL
# ==========================================
def apply_wmo_style(fig, title_text, x_label, y_label):
    fig.update_layout(
        title=dict(text=f"<b>{title_text}</b>", font=dict(size=18, color="#0B3C5D")),
        xaxis_title=f"<b>{x_label}</b>", 
        yaxis_title=f"<b>{y_label}</b>",
        margin=dict(l=40, r=20, t=60, b=100), 
        template="plotly_white", 
        hovermode="x unified",
        legend=dict(title="<b>Indikator Data:</b>", orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        plot_bgcolor="rgba(248, 249, 250, 0.5)"
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.5)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.5)')
    return fig

st.markdown(f"### 📊 Visualisasi Data {param_options[selected_param].split('. ')[1]}")
st.markdown("---")

# KELOMPOK 1: DATA SYNOPTIC (Tampil Semua Kolom yg Tersedia, Termasuk Max & Min)
if selected_param in ["TempMaxMin", "RH"]:
    df_filtered = filter_df(data[selected_param])
    if df_filtered.empty: 
        st.warning(f"⚠️ Data {selected_param} kosong/tidak valid.")
    else:
        y_label = "Suhu Udara (°C)" if selected_param == "TempMaxMin" else "Kelembapan Relatif (%)"
        param_name = param_options[selected_param].split(". ")[1]
        
        plot_cols = ['00', '03', '06', '09', '12', '15', '18', '21', 'Daily_Mean', 'Max', 'Min']
        avail_cols = [c for c in plot_cols if c in df_filtered.columns]
        
        agg_df = df_filtered.groupby('Tanggal')[avail_cols].mean().reset_index().sort_values('Tanggal')
        melted = agg_df.melt(id_vars='Tanggal', value_vars=avail_cols, var_name='Variabel Observasi', value_name='Nilai')
        
        fig_line = px.line(
            melted, x='Tanggal', y='Nilai', color='Variabel Observasi', 
            markers=True, color_discrete_sequence=px.colors.qualitative.Alphabet
        )
        fig_line.update_traces(line=dict(width=2), marker=dict(size=6))
        fig_line = apply_wmo_style(fig_line, f"Grafik Harian {param_name} - {month_choice} ({selected_year})", "Tanggal", y_label)
        
        # PERBAIKAN TANGGAL: Mengunci Sumbu X murni hanya dari 1 hingga maksimal akhir bulan (menghapus 0 dan 32)
        max_tanggal = int(agg_df['Tanggal'].max()) if not agg_df.empty else 31
        fig_line.update_layout(
            xaxis=dict(
                tickmode='array', 
                tickvals=list(range(1, max_tanggal + 1)),
                range=[0.5, max_tanggal + 0.5] 
            )
        )
        st.plotly_chart(fig_line, use_container_width=True)

# KELOMPOK 2: DATA DISTRIBUSI FREKUENSI (Berdasarkan Jam)
elif selected_param in ["TempFreq", "Vis", "HS"]:
    df_filtered = filter_df(data[selected_param])
    if df_filtered.empty: 
        st.warning(f"⚠️ Data {selected_param} kosong/tidak valid.")
    else:
        if selected_param == "TempFreq":
            cols = ['5 - 0', '0 - 5', '5 - 10', '10 - 15', '15 - 20', '20 - 25', '25 - 30', '30 - 35', '> 35']
            y_label = "Persentase Kejadian Suhu (%)"
        elif selected_param == "Vis":
            cols = ['<200', '<400', '<600', '<800', '<1500', '<1800', '<3000', '<5000', '<8000']
            y_label = "Persentase Jarak Pandang (%)"
        else:
            cols = ['<150', '<200', '<300', '<500', '<1000', '<1500']
            y_label = "Persentase Dasar Awan (%)"
            
        avail_cols = [c for c in cols if c in df_filtered.columns]
        agg_v = df_filtered.groupby('Jam')[avail_cols].mean().reset_index().sort_values('Jam')
        hm_df = agg_v.melt(id_vars='Jam', value_vars=avail_cols, var_name='Kategori Parameter', value_name='Persentase Frekuensi (%)')
        
        fig_line = px.line(hm_df, x='Jam', y='Persentase Frekuensi (%)', color='Kategori Parameter', markers=True, color_discrete_sequence=PALET_KATEGORI)
        fig_line.update_traces(line=dict(width=3), marker=dict(size=8))
        fig_line = apply_wmo_style(fig_line, f"Distribusi Pola Waktu - {month_choice}", "Jam Observasi (UTC)", y_label)
        fig_line.update_layout(xaxis=dict(tickmode='linear', dtick=3))
        st.plotly_chart(fig_line, use_container_width=True)

# KELOMPOK 3: WIND ROSE
elif selected_param == "Wind":
    df_w = filter_df(data['Wind'])
    if df_w.empty: 
        st.warning("⚠️ Data Wind kosong/tidak valid.")
    else:
        dir_map = {'35-36-01':'N', '02-03-04':'NNE', '05-06-07':'ENE', '08-09-10':'E', '11-12-13':'ESE', '14-15-16':'SSE', '17-18-19':'S', '20-21-22':'SSW', '23-24-25':'WSW', '26-27-28':'W', '29-30-31':'WNW', '32-33-34':'NNW'}
        rose_df = df_w[~df_w['Direction'].isin(['CALM', 'VARIABLE'])].copy()
        if not rose_df.empty:
            rose_df['Arah Mata Angin'] = rose_df['Direction'].map(dir_map)
            avail_speeds = [s for s in ['1-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-45', '>45'] if s in rose_df.columns]
            melt_rose = rose_df.melt(id_vars=['Arah Mata Angin'], value_vars=avail_speeds, var_name='Kecepatan (Knot)', value_name='Frekuensi (%)')
            agg_rose = melt_rose.groupby(['Arah Mata Angin', 'Kecepatan (Knot)'])['Frekuensi (%)'].mean().reset_index()
            
            fig_polar = px.bar_polar(agg_rose, r="Frekuensi (%)", theta="Arah Mata Angin", color="Kecepatan (Knot)", color_discrete_sequence=px.colors.sequential.Plasma_r, template="plotly_white")
            fig_polar = apply_wmo_style(fig_polar, f"Mawar Angin (Wind Rose) - {month_choice}", "", "")
            
            fig_polar.update_layout(
                polar=dict(
                    angularaxis=dict(
                        direction="clockwise",  
                        rotation=90,            
                        categoryorder="array",  
                        categoryarray=['N', 'NNE', 'ENE', 'E', 'ESE', 'SSE', 'S', 'SSW', 'WSW', 'W', 'WNW', 'NNW']
                    )
                )
            )
            
            st.plotly_chart(fig_polar, use_container_width=True)
