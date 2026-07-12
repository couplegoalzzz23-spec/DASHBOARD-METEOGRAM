import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# 1. KONFIGURASI HALAMAN & ANTARMUKA (UI)
# ==========================================
st.set_page_config(
    page_title="ACS Meteorological Master Dashboard",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling untuk Atribut BMKG & Kenyamanan Visual Operasional
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
        .wmo-table { width: 100%; border-collapse: collapse; font-size: 13px; }
        .wmo-table th { background-color: #0B3C5D; color: white; padding: 6px; text-align: left; }
        .wmo-table td { padding: 6px; border-bottom: 1px solid #ddd; }
    </style>
""", unsafe_allow_html=True)

# Palet Warna Spektrum Kontras Tinggi (Anti-Bias Visual)
PALET_KATEGORI = ["#0074D9", "#FF4136", "#2ECC40", "#FF851B", "#B10DC9", "#FFDC00", "#39CCCC", "#F012BE"]
PALET_MUSIM_BAR = ["#4B0082", "#00A86B", "#007FFF", "#FF8C00"] # Juni (Indigo), Juli (Teal), Agustus (Biru Cerah)

MONTHS_ID = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"]

# Pemetaan Monsun & Zonasi Musim Standar BMKG/WMO
MUSIM_MAP = {
    "Desember": "DJF (Desember - Januari - Februari) | Monsun Asia / Musim Hujan",
    "Januari": "DJF (Desember - Januari - Februari) | Monsun Asia / Musim Hujan",
    "Februari": "DJF (Desember - Januari - Februari) | Monsun Asia / Musim Hujan",
    "Maret": "MAM (Maret - April - Mei) | Peralihan I / Pancaroba",
    "April": "MAM (Maret - April - Mei) | Peralihan I / Pancaroba",
    "Mei": "MAM (Maret - April - Mei) | Peralihan I / Pancaroba",
    "Juni": "JJA (Juni - Juli - Agustus) | Monsun Australia / Kemarau",
    "Juli": "JJA (Juni - Juli - Agustus) | Monsun Australia / Kemarau",
    "Agustus": "JJA (Juni - Juli - Agustus) | Monsun Australia / Kemarau",
    "September": "SON (September - Oktober - November) | Peralihan II / Pancaroba",
    "Oktober": "SON (September - Oktober - November) | Peralihan II / Pancaroba",
    "November": "SON (September - Oktober - November) | Peralihan II / Pancaroba"
}

SEKTOR_DETIL_MAP = {
    "35-36-01": "35-36-01 (N) | Utara [350°-10°]",
    "02-03-04": "02-03-04 (NNE) | Utara-Timur Laut [20°-40°]",
    "05-06-07": "05-06-07 (ENE) | Timur-Timur Laut [50°-70°]",
    "08-09-10": "08-09-10 (E) | Timur [80°-100°]",
    "11-12-13": "11-12-13 (ESE) | Timur-Tenggara [110°-130°]",
    "14-15-16": "14-15-16 (SSE) | Selatan-Tenggara [140°-160°]",
    "17-18-19": "17-18-19 (S) | Selatan [170°-190°]",
    "20-21-22": "20-21-22 (SSW) | Selatan-Barat Daya [200°-220°]",
    "23-24-25": "23-24-25 (WSW) | Barat-Barat Daya [230°-250°]",
    "26-27-28": "26-27-28 (W) | Barat [260°-280°]",
    "29-30-31": "29-30-31 (WNW) | Barat-Barat Laut [290°-310°]",
    "32-33-34": "32-33-34 (NNW) | Utara-Barat Laut [320°-340°]"
}

# ==========================================
# 2. DATA PROCESSING ENGINE (ROBUST PARSERS)
# ==========================================
def parse_synoptic(df):
    valid_rows = []
    for idx, row in df.iterrows():
        try:
            val0 = str(row.iloc[0]).strip().replace(',', '.')
            val1 = str(row.iloc[1]).strip().replace(',', '.')
            if not (val0.replace('.', '', 1).isdigit() and val1.replace('.', '', 1).isdigit()):
                continue
            v0, v1 = float(val0), float(val1)
            rest_vals = [str(x).replace(',', '.') if pd.notna(x) else np.nan for x in row.iloc[2:].values]
            if 2000 <= v0 <= 2100 and 1 <= v1 <= 31:
                valid_rows.append([int(v0), int(v1)] + rest_vals)
            elif 1 <= v0 <= 31 and 2000 <= v1 <= 2100:
                valid_rows.append([int(v1), int(v0)] + rest_vals)
        except Exception:
            continue
    if not valid_rows: return pd.DataFrame()
    parsed_df = pd.DataFrame(valid_rows)
    base_cols = ["Tahun", "Tanggal", "00", "03", "06", "09", "12", "15", "18", "21", "Daily_Mean", "Max", "Min"]
    parsed_df = parsed_df.iloc[:, :len(base_cols)]
    parsed_df.columns = base_cols[:len(parsed_df.columns)]
    for c in parsed_df.columns:
        parsed_df[c] = pd.to_numeric(parsed_df[c], errors='coerce')
    return parsed_df

def parse_hourly_freq(df, col_names):
    valid_data = []
    for idx, row in df.iterrows():
        try:
            val0 = str(row.iloc[0]).strip().replace(',', '.')
            val1 = str(row.iloc[1]).strip().replace(',', '.')
            if not (val0.replace('.', '', 1).isdigit() and val1.replace('.', '', 1).isdigit()):
                continue
            v0, v1 = float(val0), float(val1)
            rest_vals = [str(x).replace(',', '.') if pd.notna(x) else np.nan for x in row.iloc[2:].values]
            if 0 <= v0 <= 23 and 2000 <= v1 <= 2100:
                valid_data.append([int(v1), int(v0)] + rest_vals)
            elif 2000 <= v0 <= 2100 and 0 <= v1 <= 23:
                valid_data.append([int(v0), int(v1)] + rest_vals)
        except Exception:
            continue
    if not valid_data: return pd.DataFrame()
    parsed_df = pd.DataFrame(valid_data).iloc[:, :len(col_names)]
    parsed_df.columns = col_names[:len(parsed_df.columns)]
    for c in parsed_df.columns:
        parsed_df[c] = pd.to_numeric(parsed_df[c], errors='coerce')
    return parsed_df

def parse_wind(df):
    valid_data = []
    current_year = 2021
    wind_sectors = list(SEKTOR_DETIL_MAP.keys())
    
    for idx, row in df.iterrows():
        try:
            val0 = str(row.iloc[0]).strip()
            val1 = str(row.iloc[1]).strip().replace(" ", "").upper()
            val2_str = str(row.iloc[2]).strip().replace(" ", "").upper() if len(row) > 2 else ""
            
            if val0.isdigit() and len(val0) == 4 and 2000 <= int(val0) <= 2100:
                current_year = int(val0)
                
            target_dir, start_col_idx = None, 2
            if val1 in ["CALM", "VARIABLE"] or val1 in wind_sectors:
                target_dir, start_col_idx = val1, 2
            elif val2_str in ["CALM", "VARIABLE"] or val2_str in wind_sectors:
                target_dir, start_col_idx = val2_str, 3
                
            if target_dir:
                yr = int(val0) if (val0.isdigit() and len(val0) == 4 and 2000 <= int(val0) <= 2100) else current_year
                vals = [str(x).replace(',', '.') if pd.notna(x) else np.nan for x in row.iloc[start_col_idx:].values]
                valid_data.append([yr, target_dir] + vals)
        except Exception:
            continue
            
    if not valid_data: return pd.DataFrame()
    parsed_df = pd.DataFrame(valid_data)
    expected_cols = ["Tahun", "Direction", "1-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-45", ">45", "Total"]
    parsed_df = parsed_df.iloc[:, :len(expected_cols)]
    parsed_df.columns = expected_cols[:len(parsed_df.columns)]
    for c in parsed_df.columns[2:]:
        parsed_df[c] = pd.to_numeric(parsed_df[c], errors='coerce')
    return parsed_df

@st.cache_data(show_spinner=False)
def load_all_data():
    datasets = {k: pd.DataFrame() for k in ["TempMaxMin", "TempFreq", "RH", "HS", "Vis", "Wind"]}
    get_path = lambda f: f if os.path.exists(f) else (os.path.join("data", f) if os.path.exists(os.path.join("data", f)) else f)
    
    files = {
        "TempMaxMin": (get_path("TempMaxMin_2021-2025.xlsx"), parse_synoptic),
        "TempFreq": (get_path("Temp_2021-2025.xlsx"), lambda df: parse_hourly_freq(df, ["Tahun", "Jam", "5 - 0", "0 - 5", "5 - 10", "10 - 15", "15 - 20", "20 - 25", "25 - 30", "30 - 35", "> 35"])),
        "RH": (get_path("RH_2021-2025.xlsx"), parse_synoptic),
        "HS": (get_path("HS_2021-2025.xlsx"), lambda df: parse_hourly_freq(df, ["Tahun", "Jam", "<150", "<200", "<300", "<500", "<1000", "<1500"])),
        "Vis": (get_path("Vis_2021-2025.xlsx"), lambda df: parse_hourly_freq(df, ["Tahun", "Jam", "<200", "<400", "<600", "<800", "<1500", "<1800", "<3000", "<5000", "<8000"])),
        "Wind": (get_path("Wind_2021-2025.xlsx"), parse_wind)
    }
    
    for key, (filepath, parser) in files.items():
        if os.path.exists(filepath):
            all_sheets = []
            try:
                xls = pd.ExcelFile(filepath, engine='openpyxl')
                for m_idx, m_name in enumerate(MONTHS_ID):
                    matched_sheet = next((s for s in xls.sheet_names if s.strip().lower().startswith(m_name.lower()[:3]) or f"{m_idx+1:02d}" in s), None)
                    if matched_sheet:
                        df_p = parser(pd.read_excel(xls, sheet_name=matched_sheet, header=None))
                        if not df_p.empty:
                            df_p["Bulan"], df_p["Bulan_Idx"] = m_name, m_idx + 1
                            all_sheets.append(df_p)
                if all_sheets: datasets[key] = pd.concat(all_sheets, ignore_index=True)
            except Exception as e:
                st.sidebar.error(f"⚠️ Gagal load {filepath}: {str(e)}")
    return datasets

with st.spinner("🔄 Sinkronisasi Basis Data Alfanumerik Klasifikasi WMO..."):
    data = load_all_data()

# ==========================================
# 3. SIDEBAR NAVIGATION CONTROLS
# ==========================================
st.sidebar.markdown("### 🧭 Panel Kendali Klimatologi")
st.sidebar.markdown("---")
param_options = {
    "TempMaxMin": "1. Suhu Udara Synoptic (°C)", "TempFreq": "2. Distribusi Frekuensi Suhu (%)",
    "RH": "3. Kelembapan Relatif / RH (%)", "Vis": "4. Jarak Pandang / Visibility (%)",
    "HS": "5. Cloud Ceiling / Tinggi Awan (%)", "Wind": "6. Sirkulasi Wind & Analisis Monsun"
}
selected_param = st.sidebar.selectbox("Parameter Meteorologi:", list(param_options.keys()), format_func=lambda x: param_options[x])
month_choice = st.sidebar.selectbox("Filter Analisis Bulan:", ["Semua Bulan"] + MONTHS_ID)
selected_year = st.sidebar.selectbox("Filter Analisis Tahun:", ["Semua Tahun"] + [2021, 2022, 2023, 2024, 2025])

def filter_df(df, ignore_month=False):
    if df.empty: return df
    res = df.copy()
    if not ignore_month and month_choice != "Semua Bulan": res = res[res["Bulan"] == month_choice]
    if selected_year != "Semua Tahun": res = res[res["Tahun"] == int(selected_year)]
    return res

# ==========================================
# 4. TATA LETAK GRAFIK (ANTI-OVERLAPPING STYLING)
# ==========================================
def apply_wmo_style(fig, title_text, x_label, y_label):
    fig.update_layout(
        title=dict(text=f"<b>{title_text}</b>", font=dict(size=17, color="#0B3C5D")),
        xaxis_title=f"<b>{x_label}</b>", yaxis_title=f"<b>{y_label}</b>",
        margin=dict(l=50, r=30, t=70, b=140), # Margin bawah diperluas mencegah benturan teks legenda
        template="plotly_white", hovermode="x unified",
        legend=dict(
            title="<b>Komponen Data:</b>", orientation="h",
            yanchor="top", y=-0.38, xanchor="center", x=0.5 # Koordinat diturunkan agar menjauhi teks label sumbu
        ),
        plot_bgcolor="rgba(242, 244, 245, 0.4)"
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(211,211,211,0.5)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(211,211,211,0.5)")
    return fig

def create_wind_rose_figure(rose_df, title_text):
    dir_map = {k: v.split("(")[1].split(")")[0] for k, v in SEKTOR_DETIL_MAP.items()}
    df_clean = rose_df[~rose_df["Direction"].isin(["CALM", "VARIABLE"])].copy()
    if df_clean.empty: return None
    
    df_clean["Arah Mata Angin"] = df_clean["Direction"].map(dir_map)
    speeds = ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-45", ">45"]
    avail_speeds = [s for s in speeds if s in df_clean.columns]
    
    melt_rose = df_clean.melt(id_vars=["Arah Mata Angin"], value_vars=avail_speeds, var_name="Kecepatan (Knot)", value_name="Frekuensi (%)")
    agg_rose = melt_rose.groupby(["Arah Mata Angin", "Kecepatan (Knot)"])["Frekuensi (%)"].mean(numeric_only=True).reset_index()
    
    fig_polar = px.bar_polar(
        agg_rose, r="Frekuensi (%)", theta="Arah Mata Angin", color="Kecepatan (Knot)",
        color_discrete_sequence=["#001f3f", "#0074D9", "#2ECC40", "#FFDC00", "#FF851B", "#FF4136", "#F012BE", "#B10DC9", "#111111"],
        category_orders={"Kecepatan (Knot)": speeds}, template="plotly_white"
    )
    fig_polar = apply_wmo_style(fig_polar, title_text, "", "")
    fig_polar.update_layout(
        polar=dict(
            angularaxis=dict(direction="clockwise", rotation=90, categoryorder="array", categoryarray=list(dir_map.values())),
            radialaxis=dict(showgrid=True, gridcolor="rgba(211, 211, 211, 0.6)", ticksuffix="%")
        )
    )
    return fig_polar

# DISPLAY UTAMA GRAPH INTERACTION
st.markdown(f"### 📊 Dashboard Analisis: {param_options[selected_param].split('. ')[1]}")
st.markdown("---")

if selected_param in ["TempMaxMin", "RH"]:
    df_filtered = filter_df(data[selected_param])
    if df_filtered.empty:
        st.warning("⚠️ Data kosong pada filter terpilih.")
    else:
        y_lbl = "Suhu Udara (°C)" if selected_param == "TempMaxMin" else "Kelembapan Relatif (%)"
        cols = ["00", "03", "06", "09", "12", "15", "18", "21", "Daily_Mean", "Max", "Min"]
        avail_cols = [c for c in cols if c in df_filtered.columns]
        agg_df = df_filtered.groupby("Tanggal")[avail_cols].mean(numeric_only=True).reset_index().sort_values("Tanggal")
        melted = agg_df.melt(id_vars="Tanggal", value_vars=avail_cols, var_name="Jam / Indikator", value_name="Nilai")
        
        fig_line = px.line(melted, x="Tanggal", y="Nilai", color="Jam / Indikator", markers=True, color_discrete_sequence=PALET_KATEGORI)
        fig_line.update_traces(line=dict(width=2), marker=dict(size=6), hovertemplate="<b>%{y:.1f}</b>")
        fig_line = apply_wmo_style(fig_line, f"Trend Harian Real-Time - {month_choice} ({selected_year})", "Tanggal Pengamatan", y_lbl)
        fig_line.update_layout(xaxis=dict(tickmode="linear", dtick=1, range=[0.5, int(agg_df["Tanggal"].max())+0.5]))
        st.plotly_chart(fig_line, use_container_width=True)

elif selected_param in ["TempFreq", "Vis", "HS"]:
    df_filtered = filter_df(data[selected_param])
    if df_filtered.empty:
        st.warning("⚠️ Data tidak ditemukan untuk filter ini.")
    else:
        if selected_param == "TempFreq":
            cols = ["5 - 0", "0 - 5", "5 - 10", "10 - 15", "15 - 20", "20 - 25", "25 - 30", "30 - 35", "> 35"]
            y_lbl = "Persentase Distribusi Suhu (%)"
        elif selected_param == "Vis":
            cols = ["<200", "<400", "<600", "<800", "<1500", "<1800", "<3000", "<5000", "<8000"]
            y_lbl = "Persentase Jarak Pandang (%)"
        else:
            cols = ["<150", "<200", "<300", "<500", "<1000", "<1500"]
            y_lbl = "Persentase Tinggi Dasar Awan (%)"
            
        avail_cols = [c for c in cols if c in df_filtered.columns]
        agg_v = df_filtered.groupby("Jam")[avail_cols].mean(numeric_only=True).reset_index().sort_values("Jam")
        hm_df = agg_v.melt(id_vars="Jam", value_vars=avail_cols, var_name="Kategori Batas", value_name="Persentase")
        
        fig_freq = px.line(hm_df, x="Jam", y="Persentase", color="Kategori Batas", markers=True, color_discrete_sequence=PALET_KATEGORI)
        fig_freq.update_traces(line=dict(width=2.5), marker=dict(size=7), hovertemplate="<b>%{y:.2f}%</b>")
        fig_freq = apply_wmo_style(fig_freq, f"Pola Distribusi Per Jam Observasi (UTC) - {month_choice}", "Jam Synoptic (UTC)", y_lbl)
        fig_freq.update_layout(xaxis=dict(tickmode="linear", dtick=3))
        st.plotly_chart(fig_freq, use_container_width=True)

elif selected_param == "Wind":
    df_w = filter_df(data["Wind"])
    if df_w.empty:
        st.warning("⚠️ Berkas Wind Rose kosong atau format kolom tidak sesuai.")
    else:
        tab_rose, tab_musim = st.tabs(["🧭 1. Spektrum Wind Rose (Umum)", "🌦️ 2. Integrasi Analisis Monsun & Sektor WMO 3600"])
        
        with tab_rose:
            fig_rose = create_wind_rose_figure(df_w, f"Mawar Angin Standar Penerbangan - {month_choice} ({selected_year})")
            if fig_rose: st.plotly_chart(fig_rose, use_container_width=True)
            else: st.info("💡 Tidak ada komponen angin terdeteksi selain data CALM/VARIABLE.")
            
        with tab_musim:
            # INTEGRASI TINGKAT TINGGI: Panduan Tabel Kamus WMO di Dalam UI Dashboard
            with st.expander("📖 PANDUAN REFERENSI DE-KODING SEKTOR ARAH ANGIN WMO CODE TABLE 3600"):
                st.markdown("""
                Sistem di bawah ini menerjemahkan pembagian **360° lingkaran kompas** menjadi **12 Sektor Utama Internasional** (lebar masing-masing sektor adalah 30°). 
                Gunakan tabel ini untuk validasi cepat arah datangnya angin terhadap posisi landas pacu pangkalan udara:
                """)
                html_table = "<table class='wmo-table'><tr><th>Kode Sektor</th><th>Mata Angin Internasional</th><th>Arti Pengamatan</th><th>Arah Sudut Kompas (Derajat)</th></tr>"
                for code, details in SEKTOR_DETIL_MAP.items():
                    parts = details.split("|")
                    lbl = parts[0].split("(")[1].replace(")", "").strip()
                    desc = parts[0].split(")")[1].strip()
                    deg = parts[1].strip()
                    html_table += f"<tr><td><b>{code}</b></td><td>{lbl}</td><td>Angin Datang Dari {desc}</td><td><code>{deg}</code></td></tr>"
                html_table += "</table>"
                st.markdown(html_table, unsafe_allow_html=True)
                st.caption("Sumber: WMO Manual on Codes No. 306 - Code Table 3600 / ICAO Annex 3 Meteorological Service.")
            
            st.markdown("<br>", unsafe_allow_html=True)
            pilihan_musim = st.selectbox("Pilih Siklus Kelompok Musim (BMKG ZOM):", sorted(list(set(MUSIM_MAP.values()))))
            
            bulan_musim = [k for k, v in MUSIM_MAP.items() if v == pilihan_musim]
            df_wind_all = data["Wind"].copy()
            if selected_year != "Semua Tahun": 
                df_wind_all = df_wind_all[df_wind_all["Tahun"] == int(selected_year)]
            df_musim = df_wind_all[df_wind_all["Bulan"].isin(bulan_musim)]
            
            if not df_musim.empty:
                col_r, col_b = st.columns([1.0, 1.2])
                
                with col_r:
                    fig_rm = create_wind_rose_figure(df_musim, f"Wind Rose Profil Musim {pilihan_musim.split(' ')[0]}")
                    if fig_rm: st.plotly_chart(fig_rm, use_container_width=True)
                    
                with col_b:
                    df_g_musim = df_musim[~df_musim["Direction"].isin(["CALM", "VARIABLE"])].groupby(["Direction", "Bulan", "Bulan_Idx"])["Total"].mean(numeric_only=True).reset_index().sort_values("Bulan_Idx")
                    
                    # MAGIC INTEGRASI: Menyuntikkan kamus detail arah langsung ke label data agar terbaca pada grafik
                    df_g_musim["Sektor_Deskriptif"] = df_g_musim["Direction"].map(SEKTOR_DETIL_MAP).fillna(df_g_musim["Direction"])
                    urutan_wmo_labels = [SEKTOR_DETIL_MAP[k] for k in SEKTOR_DETIL_MAP.keys() if SEKTOR_DETIL_MAP[k] in df_g_musim["Sektor_Deskriptif"].values]
                    
                    # Render Climatological Bar dengan Warna Kontras Tinggi
                    fig_m_bar = px.bar(
                        df_g_musim, x="Sektor_Deskriptif", y="Total", color="Bulan",
                        barmode="group", color_discrete_sequence=PALET_MUSIM_BAR,
                        category_orders={"Sektor_Deskriptif": list(SEKTOR_DETIL_MAP.values()), "Bulan": bulan_musim}
                    )
                    fig_m_bar = apply_wmo_style(fig_m_bar, f"Variabilitas Bulanan Sektor WMO 3600 ({pilihan_musim.split(' ')[0]})", "Sektor Arah Angin & Sudut Kompas WMO 3600", "Persentase Kejadian / Frekuensi (%)")
                    
                    # Pastikan penamaan kategori sumbu X terkunci rapat
                    fig_m_bar.update_xaxes(type="category", categoryorder="array", categoryarray=list(SEKTOR_DETIL_MAP.values()))
                    st.plotly_chart(fig_m_bar, use_container_width=True)
                    
                c_val = df_musim[df_musim["Direction"] == "CALM"]["Total"].mean()
                st.markdown(f"""
                    <div style="background-color: rgba(11, 60, 93, 0.08); padding: 14px 20px; border-radius: 8px; border-left: 5px solid #0B3C5D; color: #1D2731;">
                        📌 <b>Catatan Operasional Pangkalan:</b> Rata-rata persentase kecepatan angin di bawah 1 Knot (<b>Calm Wind</b>) pada periode sirkulasi 
                        <code>{pilihan_musim.split('|')[0].strip()}</code> adalah sebesar <b>{c_val:.2f}%</b>.
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Data untuk sirkulasi musim ini tidak ditemukan pada berkas Excel Anda.")
