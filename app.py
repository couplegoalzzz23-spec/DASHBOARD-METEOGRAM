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
    "#FBC02D",  # Kuning Emas (agar terbaca di background putih)
    "#388E3C",  # Hijau
    "#1976D2",  # Biru
    "#303F9F",  # Nila / Indigo
    "#7B1FA2",  # Ungu / Violet
    "#C2185B",  # Merah Muda-Ungu (Ekstra untuk parameter ke-8/9)
    "#0097A7",  # Biru Sian (Ekstra)
    "#455A64"   # Abu Slaty (Ekstra)
]

MONTHS_ID = [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni", 
    "Juli", "Agustus", "September", "Oktober", "November", "Desember"
]

# ==========================================
# 2. SMART DATA PARSERS (TAHAN BANTING)
# ==========================================
def parse_hourly_freq(df, col_names):
    """Membaca tabel frekuensi per jam (Visibilitas, Cloud Base)."""
    valid_data = []
    for idx, row in df.iterrows():
        try:
            val0 = str(row.iloc[0]).strip()
            val1 = str(row.iloc[1]).strip()
            if val0.replace('.', '', 1).isdigit() and val1.isdigit():
                hr = float(val0)
                yr = float(val1)
                if 0 <= hr <= 23 and 2000 < yr < 2100:
                    valid_data.append(row.values[:len(col_names)])
        except Exception:
            continue
    if not valid_data:
        return pd.DataFrame()
    parsed_df = pd.DataFrame(valid_data)
    parsed_df = parsed_df.iloc[:, :len(col_names)]
    parsed_df.columns = col_names
    for c in col_names:
        parsed_df[c] = pd.to_numeric(parsed_df[c], errors='coerce').fillna(0)
    return parsed_df

def parse_3hourly(df, is_temp_simple=False):
    """Membaca matriks synoptic 3-jam-an (RH, Temp, & TempMaxMin)."""
    valid_data = []
    for idx, row in df.iterrows():
        try:
            val0 = str(row.iloc[0]).strip()
            val1 = str(row.iloc[1]).strip()
            if val0.isdigit() and val1.isdigit():
                yr = float(val0)
                dt = float(val1)
                if 2000 < yr < 2100 and 1 <= dt <= 31:
                    valid_data.append(row.values)
        except Exception:
            continue
    if not valid_data:
        return pd.DataFrame()
    
    # Standar 8 Waktu Synoptic WMO: 00, 03, 06, 09, 12, 15, 18, 21 UTC
    base_cols = ['Tahun', 'Tanggal', '00', '03', '06', '09', '12', '15', '18', '21']
    if is_temp_simple or len(df.columns) < 13:
        cols = base_cols
        parsed_df = pd.DataFrame(valid_data).iloc[:, :len(cols)]
    else:
        cols = base_cols + ['Mean', 'Max', 'Min']
        parsed_df = pd.DataFrame(valid_data).iloc[:, :13]
        
    parsed_df.columns = cols[:len(parsed_df.columns)]
    for c in parsed_df.columns:
        parsed_df[c] = pd.to_numeric(parsed_df[c], errors='coerce').fillna(0)
    return parsed_df

def parse_wind(df):
    """Membaca tabel Wind Rose 30-sektor standar WMO."""
    valid_data = []
    current_year = 2021
    wind_sectors = ['35-36-01', '02-03-04', '05-06-07', '08-09-10', '11-12-13', '14-15-16', 
                    '17-18-19', '20-21-22', '23-24-25', '26-27-28', '29-30-31', '32-33-34']
    
    for idx, row in df.iterrows():
        try:
            val0 = str(row.iloc[0]).strip()
            val1 = str(row.iloc[1]).strip().replace(' ', '').upper()
            
            if val0.isdigit() and len(val0) == 4:
                current_year = int(val0)
                
            if val1 in ['CALM', 'VARIABLE'] or val1 in wind_sectors:
                yr = int(val0) if val0.isdigit() and len(val0) == 4 else current_year
                new_row = [yr, val1] + list(row.iloc[2:12].values)
                valid_data.append(new_row)
        except Exception:
            continue
            
    if not valid_data:
        return pd.DataFrame()
    parsed_df = pd.DataFrame(valid_data).iloc[:, :12]
    parsed_df.columns = ['Tahun', 'Direction', '1-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-45', '>45', 'Total']
    for c in parsed_df.columns[2:]:
        parsed_df[c] = pd.to_numeric(parsed_df[c], errors='coerce').fillna(0)
    return parsed_df

# ==========================================
# 3. ENGINE PEMUAT DATA (6 FILE LENGKAP)
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
    
    # Deteksi otomatis jalur file (mendukung root atau folder data/)
    def get_file_path(filename):
        if os.path.exists(filename): return filename
        if os.path.exists(os.path.join('data', filename)): return os.path.join('data', filename)
        return filename

    files = {
        'Temp': (get_file_path('Temp_2021-2025.xlsx'), lambda df: parse_3hourly(df, is_temp_simple=True)),
        'TempMaxMin': (get_file_path('TempMaxMin_2021-2025.xlsx'), lambda df: parse_3hourly(df, is_temp_simple=False)),
        'RH': (get_file_path('RH_2021-2025.xlsx'), lambda df: parse_3hourly(df, is_temp_simple=False)),
        'HS': (get_file_path('HS_2021-2025.xlsx'), lambda df: parse_hourly_freq(df, ['Jam', 'Tahun', '<150', '<200', '<300', '<500', '<1000', '<1500'])),
        'Vis': (get_file_path('Vis_2021-2025.xlsx'), lambda df: parse_hourly_freq(df, ['Jam', 'Tahun', '<200', '<400', '<600', '<800', '<1500', '<1800', '<3000', '<5000', '<8000'])),
        'Wind': (get_file_path('Wind_2021-2025.xlsx'), parse_wind)
    }
    
    for key, (filepath, parser) in files.items():
        if os.path.exists(filepath):
            all_sheets = []
            try:
                xls = pd.ExcelFile(filepath, engine='openpyxl')
                for month_idx, sheet in enumerate(MONTHS_ID):
                    if sheet in xls.sheet_names:
                        df_raw = pd.read_excel(xls, sheet_name=sheet, header=None)
                        df_parsed = parser(df_raw)
                        if not df_parsed.empty:
                            df_parsed['Bulan'] = sheet
                            df_parsed['Bulan_Idx'] = month_idx + 1
                            all_sheets.append(df_parsed)
                if all_sheets:
                    datasets[key] = pd.concat(all_sheets, ignore_index=True)
            except Exception as e:
                st.sidebar.error(f"⚠️ Gagal membaca {filepath}: {str(e)}")
                
    return datasets

with st.spinner("🔄 Mengekstrak 6 File Laporan Climatological WMO..."):
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

# Pilihan Tampilan Visualisasi (Meteogram vs Histogram Diurnal)
col_ctrl1, col_ctrl2 = st.columns([3, 1])
with col_ctrl2:
    chart_type = st.radio("Pilih Mode Visualisasi:", ["📈 Meteogram (Kurva Garis)", "📊 Histogram Diurnal (Batang)"], horizontal=False)

# ---- PARAMETER 1: SUHU UDARA SYNOPTIC (Temp) ----
if selected_param == "Temp":
    df_temp = filter_df(data['Temp'])
    if df_temp.empty:
        st.warning("⚠️ Data file `Temp_2021-2025.xlsx` kosong atau belum diunggah di repositori GitHub.")
    else:
        syn_hours = ['00', '03', '06', '09', '12', '15', '18', '21']
        avail_hours = [h for h in syn_hours if h in df_temp.columns]
        df_melt = df_temp.melt(id_vars=['Tahun', 'Bulan_Idx', 'Tanggal'], value_vars=avail_hours, var_name='Jam_UTC', value_name='Suhu')
        agg_df = df_melt.groupby('Jam_UTC')['Suhu'].mean().reset_index().sort_values('Jam_UTC')
        
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
        st.warning("⚠️ Data file `TempMaxMin_2021-2025.xlsx` tidak ditemukan atau format tidak valid.")
    else:
        syn_hours = ['00', '03', '06', '09', '12', '15', '18', '21']
        avail_hours = [h for h in syn_hours if h in df_t.columns]
        df_melt = df_t.melt(id_vars=['Tahun', 'Bulan_Idx', 'Tanggal'], value_vars=avail_hours, var_name='Jam_UTC', value_name='Suhu')
        agg_df = df_melt.groupby('Jam_UTC')['Suhu'].mean().reset_index().sort_values('Jam_UTC')
        
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
        syn_hours = ['00', '03', '06', '09', '12', '15', '18', '21']
        avail_hours = [h for h in syn_hours if h in df_r.columns]
        df_melt = df_r.melt(id_vars=['Tahun', 'Bulan_Idx', 'Tanggal'], value_vars=avail_hours, var_name='Jam_UTC', value_name='RH')
        agg_df = df_melt.groupby('Jam_UTC')['RH'].mean().reset_index().sort_values('Jam_UTC')
        
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
        
        # Urutan baku kompas meteorologi WMO (Searah jarum jam dari Utara)
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
            
            # Kepatuhan WMO: Putar sumbu 90 derajat agar Utara di atas, putaran searah jarum jam
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
