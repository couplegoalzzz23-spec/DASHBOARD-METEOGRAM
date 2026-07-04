import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

# ==========================================
# 1. PAGE CONFIGURATION & UI STYLING
# ==========================================
st.set_page_config(
    page_title="TNI AU Diurnal Meteorological Dashboard",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        .bmkg-header {
            background: linear-gradient(135deg, #0B3C5D 0%, #328CC1 100%);
            padding: 20px; border-radius: 10px; color: white; margin-bottom: 25px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .bmkg-header h1 { margin: 0; font-size: 28px; font-weight: 700; }
        .bmkg-header p { margin: 5px 0 0 0; font-size: 14px; opacity: 0.9; }
        div[data-testid="stMetricValue"] { color: #1E3A8A; font-size: 24px;}
    </style>
""", unsafe_allow_html=True)

MONTHS_ID = [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni", 
    "Juli", "Agustus", "September", "Oktober", "November", "Desember"
]

# ==========================================
# 2. SMART DATA PARSERS (ROBUST EXTRACTION)
# ==========================================
# Algoritma ini didesain untuk membaca format laporan legacy BMKG yang kompleks

def parse_hourly_freq(df, col_names):
    """Membaca tabel frekuensi per jam (Visibilitas, Cloud Base, Temp)."""
    valid_data = []
    for idx, row in df.iterrows():
        try:
            val0 = str(row.iloc[0]).strip()
            val1 = str(row.iloc[1]).strip()
            # Cek jika kolom 0 adalah jam (0-23) dan kolom 1 adalah tahun
            if val0.replace('.','',1).isdigit() and val1.isdigit():
                hr = float(val0)
                yr = float(val1)
                if 0 <= hr <= 23 and 2000 < yr < 2100:
                    valid_data.append(row.values)
        except Exception:
            pass
    if not valid_data: return pd.DataFrame()
    parsed_df = pd.DataFrame(valid_data).iloc[:, :len(col_names)]
    parsed_df.columns = col_names
    # Convert all except year to float
    for c in col_names:
        parsed_df[c] = pd.to_numeric(parsed_df[c], errors='coerce').fillna(0)
    return parsed_df

def parse_3hourly(df):
    """Membaca matriks synoptic 3-jam-an (RH & TempMaxMin)."""
    valid_data = []
    for idx, row in df.iterrows():
        try:
            val0 = str(row.iloc[0]).strip()
            val1 = str(row.iloc[1]).strip()
            # Kolom 0 = Tahun, Kolom 1 = Tanggal (1-31)
            if val0.isdigit() and val1.isdigit():
                yr = float(val0)
                dt = float(val1)
                if 2000 < yr < 2100 and 1 <= dt <= 31:
                    valid_data.append(row.values)
        except Exception:
            pass
    if not valid_data: return pd.DataFrame()
    parsed_df = pd.DataFrame(valid_data).iloc[:, :13]
    parsed_df.columns = ['Tahun', 'Tanggal', '00', '03', '06', '09', '12', '15', '18', '21', 'Mean', 'Max', 'Min']
    for c in parsed_df.columns:
        parsed_df[c] = pd.to_numeric(parsed_df[c], errors='coerce').fillna(0)
    return parsed_df

def parse_wind(df):
    """Membaca tabel Wind Rose 30-sektor."""
    valid_data = []
    current_year = 2021
    
    wind_sectors = ['35-36-01', '02-03-04', '05-06-07', '08-09-10', '11-12-13', '14-15-16', 
                    '17-18-19', '20-21-22', '23-24-25', '26-27-28', '29-30-31', '32-33-34']
    
    for idx, row in df.iterrows():
        val0 = str(row.iloc[0]).strip()
        val1 = str(row.iloc[1]).strip().replace(' ', '') # Hapus spasi agar seragam
        
        if val0.isdigit() and len(val0) == 4:
            current_year = int(val0)
            
        if val1 == 'CALM' or val1 == 'VARIABLE' or val1 in wind_sectors:
            yr = int(val0) if val0.isdigit() and len(val0)==4 else current_year
            new_row = [yr, val1] + list(row.iloc[2:12].values)
            valid_data.append(new_row)
            
    if not valid_data: return pd.DataFrame()
    parsed_df = pd.DataFrame(valid_data).iloc[:, :12]
    parsed_df.columns = ['Tahun', 'Direction', '1-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-45', '>45', 'Total']
    for c in parsed_df.columns[2:]:
        parsed_df[c] = pd.to_numeric(parsed_df[c], errors='coerce').fillna(0)
    return parsed_df

# ==========================================
# 3. DATA LOADER ENGINE
# ==========================================
@st.cache_data(show_spinner=False)
def load_all_data():
    datasets = {
        'RH': pd.DataFrame(), 'TempMaxMin': pd.DataFrame(),
        'HS': pd.DataFrame(), 'Vis': pd.DataFrame(), 'Wind': pd.DataFrame()
    }
    
    files = {
        'RH': ('RH_2021-2025.xlsx', parse_3hourly),
        'TempMaxMin': ('TempMaxMin_2021-2025.xlsx', parse_3hourly),
        'HS': ('HS_2021-2025.xlsx', lambda df: parse_hourly_freq(df, ['Jam', 'Tahun', '<150', '<200', '<300', '<500', '<1000', '<1500'])),
        'Vis': ('Vis_2021-2025.xlsx', lambda df: parse_hourly_freq(df, ['Jam', 'Tahun', '<200', '<400', '<600', '<800', '<1500', '<1800', '<3000', '<5000', '<8000'])),
        'Wind': ('Wind_2021-2025.xlsx', parse_wind)
    }
    
    for key, (filename, parser) in files.items():
        if os.path.exists(filename):
            all_sheets = []
            try:
                xls = pd.ExcelFile(filename, engine='openpyxl')
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
                st.error(f"Error memproses {filename}: {str(e)}")
                
    return datasets

with st.spinner("Membaca dan menstrukturisasi ulang data Excel WMO..."):
    data = load_all_data()

# ==========================================
# 4. SIDEBAR & FILTERING
# ==========================================
st.sidebar.image("https://www.bmkg.go.id/asset/img/logo/logo-TNI AU.png", width=90)
st.sidebar.markdown("<h2 style='color:white; margin-top:10px;'>Navigasi Data</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

param_options = {
    "Temperature": "Suhu Udara (°C) [Synoptic]",
    "RH": "Kelembapan Relatif (%) [Synoptic]",
    "Visibility": "Jarak Pandang (Km) [Frequencies]",
    "Cloud Base (HS)": "Tinggi Dasar Awan [Frequencies]",
    "Wind": "Mawar Angin (Wind Rose)"
}
selected_param = st.sidebar.selectbox("Pilih Parameter Utama", list(param_options.keys()), format_func=lambda x: param_options[x])
month_choice = st.sidebar.selectbox("Pilih Bulan", ["Semua Bulan"] + MONTHS_ID)
available_years = [2021, 2022, 2023, 2024, 2025]
selected_year = st.sidebar.selectbox("Pilih Tahun", ["Semua Tahun"] + available_years)

st.sidebar.markdown("---")
st.sidebar.caption("Sistem Pengekstrak Data Laporan Silang WMO otomatis dirancang untuk Dashboard Interaktif.")

def filter_df(df):
    if df.empty: return df
    res = df.copy()
    if month_choice != "Semua Bulan":
        res = res[res['Bulan'] == month_choice]
    if selected_year != "Semua Tahun":
        res = res[res['Tahun'] == int(selected_year)]
    return res

# ==========================================
# 5. HEADER & KPI METRICS
# ==========================================
st.markdown("""
    <div class="bmkg-header">
        <h1>Metac_IDAF - Analisis Diurnal & Frekuensi Cuaca</h1>
        <p>Sistem Ekstraksi Otomatis & Visualisasi Data Permukaan Historis 2021-2025</p>
    </div>
""", unsafe_allow_html=True)

kpi_cols = st.columns(4)
# Calculate quick KPIs based on available filtered data
with kpi_cols[0]:
    t_df = filter_df(data['TempMaxMin'])
    val = f"{t_df['Mean'].mean():.1f} °C" if not t_df.empty else "N/A"
    st.metric("Avg Mean Temperature", val)
with kpi_cols[1]:
    r_df = filter_df(data['RH'])
    val = f"{r_df['Mean'].mean():.1f} %" if not r_df.empty else "N/A"
    st.metric("Avg Relative Humidity", val)
with kpi_cols[2]:
    w_df = filter_df(data['Wind'])
    if not w_df.empty:
        calm_pct = w_df[w_df['Direction'] == 'CALM']['Total'].mean()
        val = f"{calm_pct:.1f} %"
    else: val = "N/A"
    st.metric("Rerata Kondisi CALM (Angin)", val)
with kpi_cols[3]:
    v_df = filter_df(data['Vis'])
    val = f"{v_df['<8000'].mean():.1f} %" if (not v_df.empty and '<8000' in v_df.columns) else "N/A"
    st.metric("Frekuensi Visibilitas < 8Km", val)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 6. VISUALIZATIONS
# ==========================================
def add_watermark(fig):
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=50), template="plotly_white")
    return fig

if selected_param == "Temperature":
    st.subheader("Pola Suhu Udara Synoptic (3-Jam-an UTC)")
    df_t = filter_df(data['TempMaxMin'])
    if df_t.empty:
        st.warning("Data Suhu belum/tidak diunggah.")
    else:
        # Melt data untuk merubah format kolom jam menjadi baris agar mudah di-plot
        df_melt = df_t.melt(id_vars=['Tahun', 'Bulan_Idx', 'Tanggal'], 
                            value_vars=['00', '03', '06', '09', '12', '15', '18', '21'],
                            var_name='Jam_UTC', value_name='Suhu')
        
        # Hitung rerata berdasarkan Jam
        agg_df = df_melt.groupby('Jam_UTC')['Suhu'].mean().reset_index()
        
        fig = px.line(agg_df, x='Jam_UTC', y='Suhu', markers=True, 
                      title=f"Kurva Suhu Diurnal - {month_choice} ({selected_year})",
                      labels={'Jam_UTC': 'Jam Synoptic (UTC)', 'Suhu': 'Temperatur (°C)'})
        fig.update_traces(line=dict(color='#E53935', width=3))
        st.plotly_chart(add_watermark(fig), use_container_width=True)

elif selected_param == "RH":
    st.subheader("Pola Kelembapan Relatif (RH) Synoptic")
    df_r = filter_df(data['RH'])
    if df_r.empty:
        st.warning("Data RH belum/tidak diunggah.")
    else:
        df_melt = df_r.melt(id_vars=['Tahun', 'Bulan_Idx', 'Tanggal'], 
                            value_vars=['00', '03', '06', '09', '12', '15', '18', '21'],
                            var_name='Jam_UTC', value_name='RH')
        agg_df = df_melt.groupby('Jam_UTC')['RH'].mean().reset_index()
        
        fig = px.line(agg_df, x='Jam_UTC', y='RH', markers=True,
                      title=f"Kurva RH Diurnal - {month_choice} ({selected_year})",
                      labels={'Jam_UTC': 'Jam Synoptic (UTC)', 'RH': 'Kelembapan (%)'})
        fig.update_traces(line=dict(color='#1E88E5', width=3))
        fig.update_layout(yaxis=dict(range=[40, 105]))
        st.plotly_chart(add_watermark(fig), use_container_width=True)

elif selected_param == "Visibility":
    st.subheader("Meteogram Fluktuasi Diurnal Jarak Pandang (Visibility)")
    df_v = filter_df(data['Vis'])
    if df_v.empty:
        st.warning("Data Visibilitas tidak ditemukan.")
    else:
        # Rata-rata persentase frekuensi per jam dan pastikan terurut jam 0-23
        cols = ['<200', '<400', '<600', '<800', '<1500', '<1800', '<3000', '<5000', '<8000']
        agg_v = df_v.groupby('Jam')[cols].mean().reset_index().sort_values('Jam')
        
        # Melt untuk grafik fluktuasi/meteogram
        hm_df = agg_v.melt(id_vars='Jam', value_vars=cols, var_name='Kategori_Vis', value_name='Frekuensi')
        
        # PERBAIKAN: Gunakan palet warna kontras tinggi yang sangat mencolok agar mudah dibaca saat garis menumpuk
        vibrant_palette = [
            "#D32F2F", "#1976D2", "#388E3C", "#F57C00", 
            "#7B1FA2", "#0097A7", "#E64A19", "#5D4037", "#111111"
        ]
        
        fig = px.line(hm_df, x='Jam', y='Frekuensi', color='Kategori_Vis', markers=True,
                      title=f"Meteogram Fluktuasi Frekuensi Jarak Pandang - {month_choice} ({selected_year})",
                      labels={'Jam': 'Jam Synoptic (UTC)', 'Frekuensi': 'Frekuensi Kejadian (%)', 'Kategori_Vis': 'Batas Visibilitas (m)'},
                      color_discrete_sequence=vibrant_palette)
        
        fig.update_traces(line=dict(width=3), marker=dict(size=7))
        fig.update_layout(
            hovermode="x unified",
            xaxis=dict(tickmode='linear', tick0=0, dtick=1, range=[-0.5, 23.5]),
            yaxis=dict(rangemode='tozero', gridcolor='#E0E0E0'),
            legend=dict(title="Kategori Visibilitas", orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
        )
        st.plotly_chart(add_watermark(fig), use_container_width=True)

elif selected_param == "Cloud Base (HS)":
    st.subheader("Meteogram Fluktuasi Diurnal Tinggi Dasar Awan (Ceiling)")
    df_hs = filter_df(data['HS'])
    if df_hs.empty:
        st.warning("Data Cloud Base tidak ditemukan.")
    else:
        cols = ['<150', '<200', '<300', '<500', '<1000', '<1500']
        agg_hs = df_hs.groupby('Jam')[cols].mean().reset_index().sort_values('Jam')
        hm_df = agg_hs.melt(id_vars='Jam', value_vars=cols, var_name='Kategori_HS', value_name='Frekuensi')
        
        # PERBAIKAN: Gunakan palet warna kontras tinggi yang sangat mencolok agar mudah dibaca saat garis menumpuk
        vibrant_palette_hs = [
            "#E65100", "#1E88E5", "#00897B", "#8E24AA", "#C2185B", "#37474F"
        ]
        
        fig = px.line(hm_df, x='Jam', y='Frekuensi', color='Kategori_HS', markers=True,
                      title=f"Meteogram Fluktuasi Frekuensi Tinggi Dasar Awan - {month_choice} ({selected_year})",
                      labels={'Jam': 'Jam Synoptic (UTC)', 'Frekuensi': 'Frekuensi Kejadian (%)', 'Kategori_HS': 'Tinggi Awan (ft)'},
                      color_discrete_sequence=vibrant_palette_hs)
        
        fig.update_traces(line=dict(width=3), marker=dict(size=7))
        fig.update_layout(
            hovermode="x unified",
            xaxis=dict(tickmode='linear', tick0=0, dtick=1, range=[-0.5, 23.5]),
            yaxis=dict(rangemode='tozero', gridcolor='#E0E0E0'),
            legend=dict(title="Kategori Ceiling (ft)", orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
        )
        st.plotly_chart(add_watermark(fig), use_container_width=True)

elif selected_param == "Wind":
    st.subheader("Mawar Angin (Wind Rose) Multi-sektor")
    df_w = filter_df(data['Wind'])
    if df_w.empty:
        st.warning("Data Wind tidak ditemukan.")
    else:
        # Konversi nama sektor agar lebih cantik
        dir_map = {
            '35-36-01': 'Utara (N)', '02-03-04': 'Timur Laut (NNE)',
            '05-06-07': 'Timur Laut (ENE)', '08-09-10': 'Timur (E)',
            '11-12-13': 'Tenggara (ESE)', '14-15-16': 'Tenggara (SSE)',
            '17-18-19': 'Selatan (S)', '20-21-22': 'Barat Daya (SSW)',
            '23-24-25': 'Barat Daya (WSW)', '26-27-28': 'Barat (W)',
            '29-30-31': 'Barat Laut (WNW)', '32-33-34': 'Barat Laut (NNW)'
        }
        
        # PERBAIKAN: Urutan baku kompas meteorologi (Searah jarum jam)
        compass_order = [
            'Utara (N)', 'Timur Laut (NNE)', 'Timur Laut (ENE)', 'Timur (E)',
            'Tenggara (ESE)', 'Tenggara (SSE)', 'Selatan (S)', 'Barat Daya (SSW)',
            'Barat Daya (WSW)', 'Barat (W)', 'Barat Laut (WNW)', 'Barat Laut (NNW)'
        ]
        
        # Filter out CALM & VARIABLE for Polar chart
        rose_df = df_w[~df_w['Direction'].isin(['CALM', 'VARIABLE'])].copy()
        
        if not rose_df.empty:
            rose_df['Direction_Label'] = rose_df['Direction'].map(dir_map)
            
            # Melt speed bins
            speed_cols = ['1-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-45', '>45']
            melted_rose = rose_df.melt(id_vars=['Direction_Label'], value_vars=speed_cols, 
                                       var_name='Speed_Knot', value_name='Frequency')
            
            # Aggregate to get means if multiple years/months selected
            agg_rose = melted_rose.groupby(['Direction_Label', 'Speed_Knot'])['Frequency'].mean().reset_index()
            # Buang frekuensi 0 agar tidak membebani render Plotly
            agg_rose = agg_rose[agg_rose['Frequency'] > 0]
            
            fig = px.bar_polar(agg_rose, r="Frequency", theta="Direction_Label", color="Speed_Knot",
                               color_discrete_sequence=px.colors.sequential.Plasma_r,
                               title="Distribusi Arah dan Kecepatan Angin (Knots)",
                               template="plotly_white")
            
            # PERBAIKAN: Konfigurasi Sumbu Polar/Angular agar sesuai standar WMO (Utara di Atas)
            fig.update_layout(
                polar=dict(
                    angularaxis=dict(
                        direction="clockwise",       # Berputar searah jarum jam
                        categoryorder="array",       # Memaksa Plotly mengikuti urutan array kita
                        categoryarray=compass_order, # Array arah yang sudah disiapkan di atas
                        rotation=90                  # Memutar sumbu agar elemen pertama (Utara) berada persis di atas (90 derajat)
                    )
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        # Tampilkan data kondisi CALM dan VARIABLE
        calm_df = df_w[df_w['Direction'].isin(['CALM', 'VARIABLE'])]
        if not calm_df.empty:
            st.caption("Catatan Kondisi Khusus:")
            st.dataframe(calm_df.groupby('Direction')['Total'].mean().reset_index().rename(columns={'Total': 'Persentase Kejadian (%)'}))
