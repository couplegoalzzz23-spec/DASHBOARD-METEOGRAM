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

# Palet Warna MEJIKUHIBINIU (Merah, Jingga, Kuning, Hijau, Biru, Nila, Ungu)
# Disesuaikan nilai hex-nya agar memiliki kontras tinggi dan memenuhi standar visibilitas grafik
PALET_MEJIKUHIBINIU = [
    "#D32F2F",  # Merah
    "#F57C00",  # Jingga / Orange
    "#FBC02D",  # Kuning Emas (agar jelas terbaca di background putih)
    "#388E3C",  # Hijau
    "#1976D2",  # Biru
    "#303F9F",  # Nila / Indigo
    "#7B1FA2",  # Ungu / Violet
    "#C2185B",  # Merah Muda-Ungu (Ekstra)
    "#0097A7",  # Biru Sian (Ekstra)
    "#455A64"   # Abu Slaty (Ekstra)
]

MONTHS_ID = [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni", 
    "Juli", "Agustus", "September", "Oktober", "November", "Desember"
]

# ==========================================
# 2. SMART DATA PARSERS (SUPER ADAPTIVE & TAHAN BANTING)
# ==========================================
def parse_synoptic_adaptive(df, is_temp=False):
    """
    Parser universal untuk membaca tabel Synoptic (8 Jam/24 Jam) maupun format tabel jam-jaman.
    Otomatis mendeteksi struktur kolom Excel dari laporan BMKG yang bervariasi.
    """
    valid_rows = []
    format_type = "synoptic_yr_dt"

    for idx, row in df.iterrows():
        try:
            val0_str = str(row.iloc[0]).strip().replace(',', '.')
            val1_str = str(row.iloc[1]).strip().replace(',', '.')
            if not (val0_str.replace('.', '', 1).isdigit() and val1_str.replace('.', '', 1).isdigit()):
                continue
            
            v0 = float(val0_str)
            v1 = float(val1_str)
            
            # Pola 1: Col 0 = Tahun (2000-2100), Col 1 = Tanggal (1-31)
            if 2000 <= v0 <= 2100 and 1 <= v1 <= 31:
                valid_rows.append([int(v0), int(v1)] + list(row.iloc[2:].values))
                format_type = "synoptic_yr_dt"
            # Pola 2: Col 0 = Tanggal (1-31), Col 1 = Tahun (2000-2100)
            elif 1 <= v0 <= 31 and 2000 <= v1 <= 2100:
                valid_rows.append([int(v1), int(v0)] + list(row.iloc[2:].values))
                format_type = "synoptic_yr_dt"
            # Pola 3: Col 0 = Jam (0-23), Col 1 = Tahun (2000-2100) -> Format Tabel Jam-jaman
            elif 0 <= v0 <= 23 and 2000 <= v1 <= 2100:
                valid_rows.append([int(v1), int(v0)] + list(row.iloc[2:].values))
                format_type = "hourly_row"
            # Pola 4: Col 0 = Tahun (2000-2100), Col 1 = Jam (0-23)
            elif 2000 <= v0 <= 2100 and 0 <= v1 <= 23:
                valid_rows.append([int(v0), int(v1)] + list(row.iloc[2:].values))
                format_type = "hourly_row"
        except Exception:
            continue
            
    if not valid_rows:
        return pd.DataFrame()
        
    parsed_df = pd.DataFrame(valid_rows)
    
    if format_type == "synoptic_yr_dt":
        num_cols = len(parsed_df.columns) - 2
        if num_cols >= 24:
            hours_cols = [f"{h:02d}" for h in range(24)]
            cols = ['Tahun', 'Tanggal'] + hours_cols
            parsed_df = parsed_df.iloc[:, :2+len(hours_cols)]
            parsed_df.columns = cols
        elif num_cols >= 8:
            base_hours = ['00', '03', '06', '09', '12', '15', '18', '21']
            if not is_temp and num_cols >= 11:
                cols = ['Tahun', 'Tanggal'] + base_hours + ['Mean', 'Max', 'Min']
                parsed_df = parsed_df.iloc[:, :len(cols)]
                parsed_df.columns = cols
            else:
                cols = ['Tahun', 'Tanggal'] + base_hours
                parsed_df = parsed_df.iloc[:, :len(cols)]
                parsed_df.columns = cols
        else:
            hours_cols = [f"{h*3:02d}" for h in range(num_cols)]
            cols = ['Tahun', 'Tanggal'] + hours_cols
            parsed_df.columns = cols
            
    elif format_type == "hourly_row":
        cols = ['Tahun', 'Jam', 'Nilai']
        if len(parsed_df.columns) >= 3:
            parsed_df = parsed_df.iloc[:, :3]
            parsed_df.columns = cols
        else:
            return pd.DataFrame()
            
    for c in parsed_df.columns:
        parsed_df[c] = pd.to_numeric(parsed_df[c], errors='coerce').fillna(0)
        
    return parsed_df

def parse_hourly_freq(df, col_names):
    """Membaca tabel frekuensi per jam (Visibilitas, Cloud Base)."""
    valid_data = []
    for idx, row in df.iterrows():
        try:
            val0_str = str(row.iloc[0]).strip().replace(',', '.')
            val1_str = str(row.iloc[1]).strip().replace(',', '.')
            if not (val0_str.replace('.', '', 1).isdigit() and val1_str.replace('.', '', 1).isdigit()):
                continue
            v0 = float(val0_str)
            v1 = float(val1_str)
            
            if 0 <= v0 <= 23 and 2000 <= v1 <= 2100:
                valid_data.append([int(v0), int(v1)] + list(row.iloc[2:].values))
            elif 2000 <= v0 <= 2100 and 0 <= v1 <= 23:
                valid_data.append([int(v1), int(v0)] + list(row.iloc[2:].values))
        except Exception:
            continue
            
    if not valid_data:
        return pd.DataFrame()
    parsed_df = pd.DataFrame(valid_data)
    parsed_df = parsed_df.iloc[:, :len(col_names)]
    parsed_df.columns = col_names[:len(parsed_df.columns)]
    for c in parsed_df.columns:
        parsed_df[c] = pd.to_numeric(parsed_df[c], errors='coerce').fillna(0)
    return parsed_df

def parse_wind(df):
    """Membaca tabel Wind Rose 30-sektor standar WMO dengan toleransi pergeseran kolom."""
    valid_data = []
    current_year = 2021
    wind_sectors = ['35-36-01', '02-03-04', '05-06-07', '08-09-10', '11-12-13', '14-15-16', 
                    '17-18-19', '20-21-22', '23-24-25', '26-27-28', '29-30-31', '32-33-34']
    
    for idx, row in df.iterrows():
        try:
            val0 = str(row.iloc[0]).strip()
            val1 = str(row.iloc[1]).strip().replace(' ', '').upper()
            val2_str = str(row.iloc[2]).strip().replace(' ', '').upper() if len(row) > 2 else ""
            
            if val0.isdigit() and len(val0) == 4 and 2000 <= int(val0) <= 2100:
                current_year = int(val0)
                
            target_dir = None
            start_col_idx = 2
            if val1 in ['CALM', 'VARIABLE'] or val1 in wind_sectors:
                target_dir = val1
                start_col_idx = 2
            elif val2_str in ['CALM', 'VARIABLE'] or val2_str in wind_sectors:
                target_dir = val2_str
                start_col_idx = 3
                
            if target_dir:
                yr = int(val0) if (val0.isdigit() and len(val0) == 4 and 2000 <= int(val0) <= 2100) else current_year
                new_row = [yr, target_dir] + list(row.iloc[start_col_idx:].values)
                valid_data.append(new_row)
        except Exception:
            continue
            
    if not valid_data:
        return pd.DataFrame()
    parsed_df = pd.DataFrame(valid_data)
    expected_cols = ['Tahun', 'Direction', '1-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-45', '>45', 'Total']
    parsed_df = parsed_df.iloc[:, :len(expected_cols)]
    parsed_df.columns = expected_cols[:len(parsed_df.columns)]
    for c in parsed_df.columns[2:]:
        parsed_df[c] = pd.to_numeric(parsed_df[c], errors='coerce').fillna(0)
    return parsed_df

# ==========================================
# 3. ENGINE PEMUAT DATA (6 FILE LENGKAP & SMART MATCHING)
# ==========================================
@st.cache_data(show_spinner=False)
def load_all_data():
    datasets = {
        'Temp': pd.DataFrame(),       # File ke-1
        'TempMaxMin': pd.DataFrame(), # File ke-2
        'RH': pd.DataFrame(),         # File ke-3
        'HS': pd.DataFrame(),         # File ke-4
        'Vis': pd.DataFrame(),        # File ke-5
        'Wind': pd.DataFrame()        # File ke-6
    }
    
    def get_file_path(filename):
        if os.path.exists(filename): return filename
        if os.path.exists(os.path.join('data', filename)): return os.path.join('data', filename)
        # Pencarian case-insensitive di direktori aktif
        for f in os.listdir('.'):
            if f.lower() == filename.lower():
                return f
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
                    # Smart Case-Insensitive Sheet Matching
                    matched_sheet = None
                    for s_name in xls.sheet_names:
                        s_clean = str(s_name).strip().lower()
                        m_clean = month_name.lower()
                        m_short = m_clean[:3]
                        if (s_clean == m_clean or s_clean.startswith(m_clean) or 
                            s_clean.startswith(m_short) or f"{month_idx+1:02d}" in s_clean):
                            matched_sheet = s_name
                            break
                    
                    if matched_sheet:
                        df_raw = pd.read_excel(xls, sheet_name=matched_sheet, header=None)
                        df_parsed = parser(df_raw)
                        if not df_parsed.empty:
                            df_parsed['Bulan'] = month_name
                            df_parsed['Bulan_Idx'] = month_idx + 1
                            all_sheets.append(df_parsed)
                if all_sheets:
                    datasets[key] = pd.concat(all_sheets, ignore_index=True)
            except Exception as e:
                st.sidebar.error(f"⚠️ Gagal memproses {filepath}: {str(e)}")
                
    return datasets

with st.spinner("🔄 Mengekstrak & Menstrukturkan 6 File Laporan Climatological WMO..."):
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

selected_param = st.sidebar.selectbox("Pilih Parameter Meteorologi (6 File):", list(param_options.keys()), format_func=lambda x: param_options[x])
month_choice = st.sidebar.selectbox("Filter Bulan:", ["Semua Bulan"] + MONTHS_ID)
available_years = [2021, 2022, 2023, 2024, 2025]
selected_year = st.sidebar.selectbox("Filter Tahun:", ["Semua Tahun"] + available_years)

st.sidebar.markdown("---")
st.sidebar.caption("💡 **Standar ICAO/WMO:** Seluruh observasi waktu menggunakan standar UTC Synoptic. Arah angin mengindikasikan arah datangnya angin.")

def filter_df(df):
    if df.empty: return df
    res = df.copy()
    if month_choice != "Semua Bulan":
        res = res[res['Bulan'] == month_choice]
    if selected_year != "Semua Tahun":
        res = res[res['Tahun'] == int(selected_year)]
    return res

# ==========================================
# 5. HEADER & METRIK UTAMA (KPI)
# ==========================================
st.markdown("""
    <div class="bmkg-header">
        <h1>Dashboard Climatological Summary (ACS) & Diurnal Analysis</h1>
        <p>Sistem Pemantauan Terintegrasi 6 Parameter Meteorologi Permukaan (Suhu, RH, Visibilitas, Ceiling, & Angin)</p>
    </div>
""", unsafe_allow_html=True)

kpi_cols = st.columns(4)
with kpi_cols[0]:
    t_df = filter_df(data['TempMaxMin'])
    val = f"{t_df['Mean'].mean():.1f} °C" if (not t_df.empty and 'Mean' in t_df.columns) else "N/A"
    st.metric("Rerata Suhu Udara", val)
with kpi_cols[1]:
    r_df = filter_df(data['RH'])
    val = f"{r_df['Mean'].mean():.1f} %" if (not r_df.empty and 'Mean' in r_df.columns) else "N/A"
    st.metric("Rerata Kelembapan (RH)", val)
with kpi_cols[2]:
    w_df = filter_df(data['Wind'])
    if not w_df.empty:
        calm_pct = w_df[w_df['Direction'] == 'CALM']['Total'].mean()
        val = f"{calm_pct:.1f} %"
    else: val = "N/A"
    st.metric("Frekuensi Angin CALM", val)
with kpi_cols[3]:
    v_df = filter_df(data['Vis'])
    val = f"{v_df['<8000'].mean():.1f} %" if (not v_df.empty and '<8000' in v_df.columns) else "N/A"
    st.metric("Visibilitas < 8000m", val)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 6. ENGINE VISUALISASI (METEOGRAM & HISTOGRAM)
# ==========================================
def apply_wmo_style(fig, title_text, x_label, y_label):
    fig.update_layout(
        title=dict(text=f"<b>{title_text}</b>", font=dict(size=18, color="#0B3C5D")),
        xaxis_title=f"<b>{x_label}</b>",
        yaxis_title=f"<b>{y_label}</b>",
        margin=dict(l=20, r=20, t=60, b=50),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            title="<b>Legenda (Mejikuhibiniu):</b>",
            orientation="v", yanchor="top", y=1, xanchor="left", x=1.02,
            bgcolor="rgba(255,255,255,0.9)", bordercolor="#E0E0E0", borderwidth=1
        )
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#F0F0F0', zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E0E0E0', zeroline=False)
    return fig

col_ctrl1, col_ctrl2 = st.columns([3, 1])
with col_ctrl2:
    chart_type = st.radio("Pilih Mode Visualisasi:", ["📈 Meteogram (Kurva Garis)", "📊 Histogram Diurnal (Batang)"], horizontal=False)

# ---- PARAMETER 1: SUHU UDARA SYNOPTIC (Temp) ----
if selected_param == "Temp":
    df_temp = filter_df(data['Temp'])
    if df_temp.empty:
        st.warning("⚠️ Data file `Temp_2021-2025.xlsx` tidak dapat diekstrak atau struktur tabel tidak valid. Cek format tabel Excel Anda.")
    else:
        if 'Jam' in df_temp.columns and 'Nilai' in df_temp.columns:
            agg_df = df_temp.groupby('Jam')['Nilai'].mean().reset_index().rename(columns={'Jam': 'Jam_UTC', 'Nilai': 'Suhu'})
            agg_df['Jam_UTC'] = agg_df['Jam_UTC'].apply(lambda x: f"{int(x):02d}")
        else:
            syn_hours = [f"{h:02d}" for h in range(24)]
            avail_hours = [h for h in syn_hours if h in df_temp.columns]
            if not avail_hours:
                avail_hours = [col for col in df_temp.columns if str(col).replace('.', '', 1).isdigit() and 0 <= float(col) <= 23]
            df_melt = df_temp.melt(id_vars=['Tahun', 'Bulan_Idx', 'Tanggal'], value_vars=avail_hours, var_name='Jam_UTC', value_name='Suhu')
            agg_df = df_melt.groupby('Jam_UTC')['Suhu'].mean().reset_index()
            agg_df['Jam_UTC'] = agg_df['Jam_UTC'].apply(lambda x: f"{int(float(str(x))):02d}")
            
        agg_df = agg_df.sort_values('Jam_UTC')
        
        if "Meteogram" in chart_type:
            fig = px.line(agg_df, x='Jam_UTC', y='Suhu', markers=True, color_discrete_sequence=[PALET_MEJIKUHIBINIU[0]])
            fig.update_traces(line=dict(width=3.5), marker=dict(size=9, color=PALET_MEJIKUHIBINIU[1]))
        else:
            fig = px.bar(agg_df, x='Jam_UTC', y='Suhu', color='Jam_UTC', color_discrete_sequence=PALET_MEJIKUHIBINIU)
            fig.update_layout(showlegend=False)
            
        fig = apply_wmo_style(fig, f"Pola Diurnal Suhu Udara Synoptic - {month_choice} ({selected_year})", "Jam Observasi Synoptic (UTC)", "Temperatur (°C)")
        st.plotly_chart(fig, use_container_width=True)

# ---- PARAMETER 2: SUHU EKSTREM (TempMaxMin) ----
elif selected_param == "TempMaxMin":
    df_t = filter_df(data['TempMaxMin'])
    if df_t.empty:
        st.warning("⚠️ Data file `TempMaxMin_2021-2025.xlsx` tidak ditemukan atau struktur tabel tidak sesuai.")
    else:
        if 'Jam' in df_t.columns and 'Nilai' in df_t.columns:
            agg_df = df_t.groupby('Jam')['Nilai'].mean().reset_index().rename(columns={'Jam': 'Jam_UTC', 'Nilai': 'Suhu'})
            agg_df['Jam_UTC'] = agg_df['Jam_UTC'].apply(lambda x: f"{int(x):02d}")
        else:
            syn_hours = [f"{h:02d}" for h in range(24)]
            avail_hours = [h for h in syn_hours if h in df_t.columns]
            if not avail_hours:
                avail_hours = [col for col in df_t.columns if str(col).replace('.', '', 1).isdigit() and 0 <= float(col) <= 23]
            df_melt = df_t.melt(id_vars=['Tahun', 'Bulan_Idx', 'Tanggal'], value_vars=avail_hours, var_name='Jam_UTC', value_name='Suhu')
            agg_df = df_melt.groupby('Jam_UTC')['Suhu'].mean().reset_index()
            agg_df['Jam_UTC'] = agg_df['Jam_UTC'].apply(lambda x: f"{int(float(str(x))):02d}")
            
        agg_df = agg_df.sort_values('Jam_UTC')
        
        if "Meteogram" in chart_type:
            fig = px.line(agg_df, x='Jam_UTC', y='Suhu', markers=True, color_discrete_sequence=[PALET_MEJIKUHIBINIU[1]])
            fig.update_traces(line=dict(width=3.5), marker=dict(size=9, color=PALET_MEJIKUHIBINIU[0]))
        else:
            fig = px.bar(agg_df, x='Jam_UTC', y='Suhu', color='Jam_UTC', color_discrete_sequence=PALET_MEJIKUHIBINIU)
            fig.update_layout(showlegend=False)
            
        fig = apply_wmo_style(fig, f"Distribusi Diurnal Suhu Udara (Dataset Ekstrem) - {month_choice} ({selected_year})", "Jam Observasi Synoptic (UTC)", "Temperatur (°C)")
        st.plotly_chart(fig, use_container_width=True)

# ---- PARAMETER 3: KELEMBAPAN RELATIF (RH) ----
elif selected_param == "RH":
    df_r = filter_df(data['RH'])
    if df_r.empty:
        st.warning("⚠️ Data file `RH_2021-2025.xlsx` tidak ditemukan.")
    else:
        if 'Jam' in df_r.columns and 'Nilai' in df_r.columns:
            agg_df = df_r.groupby('Jam')['Nilai'].mean().reset_index().rename(columns={'Jam': 'Jam_UTC', 'Nilai': 'RH'})
            agg_df['Jam_UTC'] = agg_df['Jam_UTC'].apply(lambda x: f"{int(x):02d}")
        else:
            syn_hours = [f"{h:02d}" for h in range(24)]
            avail_hours = [h for h in syn_hours if h in df_r.columns]
            if not avail_hours:
                avail_hours = [col for col in df_r.columns if str(col).replace('.', '', 1).isdigit() and 0 <= float(col) <= 23]
            df_melt = df_r.melt(id_vars=['Tahun', 'Bulan_Idx', 'Tanggal'], value_vars=avail_hours, var_name='Jam_UTC', value_name='RH')
            agg_df = df_melt.groupby('Jam_UTC')['RH'].mean().reset_index()
            agg_df['Jam_UTC'] = agg_df['Jam_UTC'].apply(lambda x: f"{int(float(str(x))):02d}")
            
        agg_df = agg_df.sort_values('Jam_UTC')
        
        if "Meteogram" in chart_type:
            fig = px.line(agg_df, x='Jam_UTC', y='RH', markers=True, color_discrete_sequence=[PALET_MEJIKUHIBINIU[4]])
            fig.update_traces(line=dict(width=3.5), marker=dict(size=9, color=PALET_MEJIKUHIBINIU[3]))
        else:
            fig = px.bar(agg_df, x='Jam_UTC', y='RH', color='Jam_UTC', color_discrete_sequence=PALET_MEJIKUHIBINIU[::-1])
            fig.update_layout(showlegend=False)
            
        fig = apply_wmo_style(fig, f"Pola Diurnal Kelembapan Relatif (RH) - {month_choice} ({selected_year})", "Jam Observasi Synoptic (UTC)", "Kelembapan (%)")
        fig.update_layout(yaxis=dict(range=[35, 105]))
        st.plotly_chart(fig, use_container_width=True)

# ---- PARAMETER 4: VISIBILITAS (Visibility) ----
elif selected_param == "Visibility":
    df_v = filter_df(data['Vis'])
    if df_v.empty:
        st.warning("⚠️ Data file `Vis_2021-2025.xlsx` tidak ditemukan.")
    else:
        cols = ['<200', '<400', '<600', '<800', '<1500', '<1800', '<3000', '<5000', '<8000']
        avail_cols = [c for c in cols if c in df_v.columns]
        agg_v = df_v.groupby('Jam')[avail_cols].mean().reset_index().sort_values('Jam')
        hm_df = agg_v.melt(id_vars='Jam', value_vars=avail_cols, var_name='Kategori_Vis', value_name='Frekuensi')
        
        if "Meteogram" in chart_type:
            fig = px.line(hm_df, x='Jam', y='Frekuensi', color='Kategori_Vis', markers=True, color_discrete_sequence=PALET_MEJIKUHIBINIU)
            fig.update_traces(line=dict(width=2.8), marker=dict(size=6))
        else:
            fig = px.bar(hm_df, x='Jam', y='Frekuensi', color='Kategori_Vis', barmode='group', color_discrete_sequence=PALET_MEJIKUHIBINIU)
            
        fig = apply_wmo_style(fig, f"Pola Diurnal Frekuensi Jarak Pandang (Visibility) - {month_choice} ({selected_year})", "Jam Observasi (UTC)", "Frekuensi Kejadian (%)")
        fig.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1, range=[-0.5, 23.5]))
        st.plotly_chart(fig, use_container_width=True)

# ---- PARAMETER 5: TINGGI DASAR AWAN (Cloud Base / HS) ----
elif selected_param == "Cloud Base (HS)":
    df_hs = filter_df(data['HS'])
    if df_hs.empty:
        st.warning("⚠️ Data file `HS_2021-2025.xlsx` tidak ditemukan.")
    else:
        cols = ['<150', '<200', '<300', '<500', '<1000', '<1500']
        avail_cols = [c for c in cols if c in df_hs.columns]
        agg_hs = df_hs.groupby('Jam')[avail_cols].mean().reset_index().sort_values('Jam')
        hm_df = agg_hs.melt(id_vars='Jam', value_vars=avail_cols, var_name='Kategori_HS', value_name='Frekuensi')
        
        if "Meteogram" in chart_type:
            fig = px.line(hm_df, x='Jam', y='Frekuensi', color='Kategori_HS', markers=True, color_discrete_sequence=PALET_MEJIKUHIBINIU)
            fig.update_traces(line=dict(width=2.8), marker=dict(size=6))
        else:
            fig = px.bar(hm_df, x='Jam', y='Frekuensi', color='Kategori_HS', barmode='group', color_discrete_sequence=PALET_MEJIKUHIBINIU)
            
        fig = apply_wmo_style(fig, f"Pola Diurnal Tinggi Dasar Awan (Ceiling / HS) - {month_choice} ({selected_year})", "Jam Observasi (UTC)", "Frekuensi Kejadian (%)")
        fig.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1, range=[-0.5, 23.5]))
        st.plotly_chart(fig, use_container_width=True)

# ---- PARAMETER 6: MAWAR ANGIN (Wind Rose) ----
elif selected_param == "Wind":
    df_w = filter_df(data['Wind'])
    if df_w.empty:
        st.warning("⚠️ Data file `Wind_2021-2025.xlsx` tidak ditemukan.")
    else:
        dir_map = {
            '35-36-01': 'Utara (N)', '02-03-04': 'Timur Laut (NNE)',
            '05-06-07': 'Timur Laut (ENE)', '08-09-10': 'Timur (E)',
            '11-12-13': 'Tenggara (ESE)', '14-15-16': 'Tenggara (SSE)',
            '17-18-19': 'Selatan (S)', '20-21-22': 'Barat Daya (SSW)',
            '23-24-25': 'Barat Daya (WSW)', '26-27-28': 'Barat (W)',
            '29-30-31': 'Barat Laut (WNW)', '32-33-34': 'Barat Laut (NNW)'
        }
        
        compass_order = [
            'Utara (N)', 'Timur Laut (NNE)', 'Timur Laut (ENE)', 'Timur (E)',
            'Tenggara (ESE)', 'Tenggara (SSE)', 'Selatan (S)', 'Barat Daya (SSW)',
            'Barat Daya (WSW)', 'Barat (W)', 'Barat Laut (WNW)', 'Barat Laut (NNW)'
        ]
        
        rose_df = df_w[~df_w['Direction'].isin(['CALM', 'VARIABLE'])].copy()
        if not rose_df.empty:
            rose_df['Direction_Label'] = rose_df['Direction'].map(dir_map)
            speed_cols = ['1-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-45', '>45']
            avail_speeds = [s for s in speed_cols if s in rose_df.columns]
            
            melted_rose = rose_df.melt(id_vars=['Direction_Label'], value_vars=avail_speeds, var_name='Speed_Knot', value_name='Frequency')
            agg_rose = melted_rose.groupby(['Direction_Label', 'Speed_Knot'])['Frequency'].mean().reset_index()
            agg_rose = agg_rose[agg_rose['Frequency'] > 0]
            
            fig = px.bar_polar(
                agg_rose, r="Frequency", theta="Direction_Label", color="Speed_Knot",
                color_discrete_sequence=PALET_MEJIKUHIBINIU,
                title=f"<b>Mawar Angin (Wind Rose) Standar WMO - {month_choice} ({selected_year})</b>",
                template="plotly_white"
            )
            
            fig.update_layout(
                polar=dict(
                    angularaxis=dict(
                        direction="clockwise",
                        categoryorder="array",
                        categoryarray=compass_order,
                        rotation=90
                    )
                ),
                legend=dict(title="<b>Kecepatan (Knots)</b>", orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)
            
        calm_df = df_w[df_w['Direction'].isin(['CALM', 'VARIABLE'])]
        if not calm_df.empty:
            st.markdown("#### 📌 Catatan Kondisi Angin Khusus")
            st.dataframe(calm_df.groupby('Direction')['Total'].mean().reset_index().rename(columns={'Total': 'Rerata Persentase Kejadian (%)'}), use_container_width=True)
