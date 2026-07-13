import os
import gc
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

# Toggle Tema di Sidebar
theme_mode = st.sidebar.radio("🌓 Mode Tampilan:", ["Dark Mode", "Light Mode"], horizontal=True)
is_dark = theme_mode == "Dark Mode"

# Warna Dinamis Berdasarkan Tema untuk Keseluruhan Aplikasi
bg_color_main = "#0E1117" if is_dark else "#FFFFFF"
text_color_main = "#FAFAFA" if is_dark else "#0F1116"
panel_bg = "#161B22" if is_dark else "#F1F5F9"
border_col = "#334155" if is_dark else "#CBD5E1"
chart_template = "plotly_dark" if is_dark else "plotly_white"

# Warna Khusus Widget & Kontras Tambahan untuk Mengatasi Blur/Gelap di Light Mode
input_bg = "#1E293B" if is_dark else "#FFFFFF"
hover_bg = "#334155" if is_dark else "#E2E8F0"

# Injeksi CSS Global untuk Menimpa Tampilan Keseluruhan Streamlit secara Presisi
st.markdown(f"""
    <style>
        /* 1. Reset Warna Background Utama dan Teks */
        .stApp, .main, .block-container {{
            background-color: {bg_color_main} !important;
            color: {text_color_main} !important;
        }}
        
        /* 2. Mengubah Warna Background & Border Sidebar */
        section[data-testid="stSidebar"] {{
            background-color: {panel_bg} !important;
            border-right: 1px solid {border_col} !important;
        }}
        section[data-testid="stSidebar"] *, section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label {{
            color: {text_color_main} !important;
        }}
        
        /* 3. Menyesuaikan Header / Top bar Aplikasi */
        header[data-testid="stHeader"] {{
            background-color: transparent !important;
        }}
        
        /* Padding container utama */
        .main .block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; }}
        
        /* 4. Penyesuaian Metrik & Alert */
        div[data-testid="stMetricValue"] {{ color: #0B3C5D !important; font-size: 24px; font-weight: bold; }}
        .stAlert {{ border-radius: 8px; }}
        
        /* 5. Tabel Custom WMO */
        .wmo-table {{ width: 100%; border-collapse: collapse; font-size: 13px; color: {text_color_main}; }}
        .wmo-table th {{ background-color: #0B3C5D; color: white; padding: 8px; text-align: left; border: 1px solid {border_col}; }}
        .wmo-table td {{ padding: 8px; border-bottom: 1px solid {border_col}; border-left: 1px solid {border_col}; border-right: 1px solid {border_col}; background-color: {bg_color_main}; color: {text_color_main}; }}
        .wmo-table tr:nth-child(even) td {{ background-color: {panel_bg}; }}
        
        /* 6. PERBAIKAN TOTAL WIDGET SELECTBOX, INPUT, & POPUP MENU (Mengatasi Tulisan Gelap di Mode Light) */
        div[data-baseweb="select"] > div {{
            background-color: {input_bg} !important;
            border: 1px solid {border_col} !important;
            color: {text_color_main} !important;
        }}
        div[data-baseweb="select"] > div * {{
            color: {text_color_main} !important;
        }}
        /* Dropdown Menu Popup (List Pilihan Saat Diklik) */
        ul[data-baseweb="menu"], div[data-baseweb="popover"], div[data-baseweb="popover"] > div {{
            background-color: {input_bg} !important;
            border: 1px solid {border_col} !important;
        }}
        ul[data-baseweb="menu"] li {{
            background-color: {input_bg} !important;
            color: {text_color_main} !important;
        }}
        ul[data-baseweb="menu"] li * {{
            color: {text_color_main} !important;
        }}
        ul[data-baseweb="menu"] li:hover, ul[data-baseweb="menu"] li:hover * {{
            background-color: {hover_bg} !important;
            color: {text_color_main} !important;
        }}
        
        /* 7. PERBAIKAN TABS STREAMLIT */
        .stTabs [data-baseweb="tab-list"] {{
            border-bottom: 2px solid {border_col} !important;
            gap: 8px;
        }}
        .stTabs [data-baseweb="tab"] {{
            background-color: {panel_bg} !important;
            border: 1px solid {border_col} !important;
            border-bottom: none !important;
            border-radius: 6px 6px 0 0 !important;
            padding: 8px 16px !important;
        }}
        .stTabs [data-baseweb="tab"] * {{
            color: {text_color_main} !important;
            opacity: 0.8 !important;
            font-weight: 600 !important;
        }}
        .stTabs [data-baseweb="tab"][aria-selected="true"] {{
            background-color: {bg_color_main} !important;
            border-top: 3px solid #0074D9 !important;
        }}
        .stTabs [data-baseweb="tab"][aria-selected="true"] * {{
            color: #0074D9 !important;
            opacity: 1 !important;
            font-weight: bold !important;
        }}
        
        /* 8. PERBAIKAN EXPANDER */
        div[data-testid="stExpander"] {{
            background-color: {panel_bg} !important;
            border: 1px solid {border_col} !important;
            border-radius: 8px !important;
        }}
        div[data-testid="stExpander"] summary, div[data-testid="stExpander"] summary * {{
            color: {text_color_main} !important;
            font-weight: 600 !important;
        }}
        div[data-testid="stExpanderDetails"] {{
            background-color: {bg_color_main} !important;
            color: {text_color_main} !important;
            border-top: 1px solid {border_col} !important;
        }}
        
        /* 9. Memastikan Teks Umum, Label Widget, & Keterangan Selalu Terbaca Jelas */
        label[data-testid="stWidgetLabel"] p, label[data-testid="stWidgetLabel"] span,
        div[data-testid="stMarkdownContainer"] p, .stMarkdown p, .stText {{
            color: {text_color_main} !important;
        }}
    </style>
""", unsafe_allow_html=True)

PALET_KATEGORI = ["#0074D9", "#FF4136", "#2ECC40", "#FF851B", "#B10DC9", "#FFDC00", "#39CCCC", "#F012BE"]
PALET_MUSIM_BAR = ["#4B0082", "#00A86B", "#007FFF", "#FF8C00"]
MONTHS_ID = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"]

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
# 2. DATA PROCESSING ENGINE
# ==========================================
def clean_num_str(val):
    if pd.isna(val): return ""
    return str(val).strip().replace(',', '.').split('.')[0]

def parse_synoptic(df):
    valid_rows = []
    for _, row in df.iterrows():
        try:
            val0 = clean_num_str(row.iloc[0])
            val1 = clean_num_str(row.iloc[1])
            if not (val0.lstrip('-').isdigit() and val1.lstrip('-').isdigit()):
                continue
            v0, v1 = int(val0), int(val1)
            rest_vals = [str(x).replace(',', '.') if pd.notna(x) else np.nan for x in row.iloc[2:].values]
            if 2000 <= v0 <= 2100 and 1 <= v1 <= 31:
                valid_rows.append([v0, v1] + rest_vals)
            elif 1 <= v0 <= 31 and 2000 <= v1 <= 2100:
                valid_rows.append([v1, v0] + rest_vals)
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
    for _, row in df.iterrows():
        try:
            val0 = clean_num_str(row.iloc[0])
            val1 = clean_num_str(row.iloc[1])
            if not (val0.lstrip('-').isdigit() and val1.lstrip('-').isdigit()):
                continue
            v0, v1 = int(val0), int(val1)
            rest_vals = [str(x).replace(',', '.') if pd.notna(x) else np.nan for x in row.iloc[2:].values]
            # Validasi jam UTC (0-23) dan konversi jika ada rekam medik yang menulis jam 24 sebagai 00 UTC
            if (0 <= v0 <= 24) and (2000 <= v1 <= 2100):
                if v0 == 24: v0 = 0
                valid_data.append([v1, v0] + rest_vals)
            elif (2000 <= v0 <= 2100) and (0 <= v1 <= 24):
                if v1 == 24: v1 = 0
                valid_data.append([v0, v1] + rest_vals)
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
    for _, row in df.iterrows():
        try:
            val0 = clean_num_str(row.iloc[0])
            val1 = str(row.iloc[1]).strip().replace(" ", "").upper() if pd.notna(row.iloc[1]) else ""
            val2_str = str(row.iloc[2]).strip().replace(" ", "").upper() if len(row) > 2 and pd.notna(row.iloc[2]) else ""
            
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
            xls = None
            try:
                xls = pd.ExcelFile(filepath, engine='openpyxl')
                for m_idx, m_name in enumerate(MONTHS_ID):
                    matched_sheet = next((s for s in xls.sheet_names if s.strip().lower().startswith(m_name.lower()[:3]) or f"{m_idx+1:02d}" in s), None)
                    if matched_sheet:
                        df_p = parser(pd.read_excel(xls, sheet_name=matched_sheet, header=None))
                        if not df_p.empty:
                            df_p["Bulan"], df_p["Bulan_Idx"] = m_name, m_idx + 1
                            all_sheets.append(df_p)
                if all_sheets: 
                    datasets[key] = pd.concat(all_sheets, ignore_index=True)
            except Exception as e:
                st.sidebar.error(f"⚠️ Gagal load {filepath}: {str(e)}")
            finally:
                if xls:
                    xls.close()
    gc.collect()
    return datasets

with st.spinner("🔄 Sinkronisasi Basis Data..."):
    data = load_all_data()

# ==========================================
# 3. SIDEBAR NAVIGATION CONTROLS
# ==========================================
st.sidebar.markdown("### 🧭 Panel Control")
st.sidebar.markdown("---")
param_options = {
    "Home": "🏠 Beranda Utama",
    "TempMaxMin": "1. Suhu Udara Synoptic (°C)", 
    "TempFreq": "2. Distribusi Frekuensi Suhu (%)",
    "RH": "3. Kelembapan Relatif / RH (%)", 
    "Vis": "4. Jarak Pandang / Visibility (%)",
    "HS": "5. HS / Tinggi Dasar Awan (%)", 
    "Wind": "6. Sirkulasi Wind & Analisis Monsun"
}

selected_param = st.sidebar.selectbox("Navigasi Sistem:", list(param_options.keys()), format_func=lambda x: param_options[x])

# Sembunyikan filter bulan dan tahun jika di menu Home
if selected_param != "Home":
    month_choice = st.sidebar.selectbox("Filter Analisis Bulan:", ["Semua Bulan"] + MONTHS_ID)
    selected_year = st.sidebar.selectbox("Filter Analisis Tahun:", ["Semua Tahun"] + [2021, 2022, 2023, 2024, 2025])

def filter_df(df, ignore_month=False):
    if df.empty: return df
    res = df.copy()
    if not ignore_month and month_choice != "Semua Bulan": res = res[res["Bulan"] == month_choice]
    if selected_year != "Semua Tahun": res = res[res["Tahun"] == int(selected_year)]
    return res

# ==========================================
# 4. TATA LETAK GRAFIK & FUNGSI BANTU
# ==========================================
def apply_wmo_style(fig, title_text, x_label, y_label):
    grid_col = "rgba(255,255,255,0.2)" if is_dark else "rgba(0,0,0,0.15)"
    font_col = "#60A5FA" if is_dark else "#0B3C5D"
    text_col = "#FAFAFA" if is_dark else "#0F1116"
    legend_bg = "rgba(14,17,23,0.85)" if is_dark else "rgba(255, 255, 255, 0.9)"
    
    fig.update_layout(
        title=dict(text=f"<b>{title_text}</b>", font=dict(size=17, color=font_col)),
        xaxis_title=dict(text=f"<b>{x_label}</b>", font=dict(color=text_col, size=14)),
        yaxis_title=dict(text=f"<b>{y_label}</b>", font=dict(color=text_col, size=14)),
        font=dict(color=text_col),
        margin=dict(l=50, r=160, t=70, b=80),
        template=chart_template, 
        hovermode="x unified",
        legend=dict(
            title=dict(text="<b>Komponen Data:</b>", font=dict(color=text_col)),
            font=dict(color=text_col),
            orientation="v", yanchor="top", y=1, xanchor="left", x=1.02,
            bgcolor=legend_bg, bordercolor=grid_col, borderwidth=1
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=grid_col, tickfont=dict(color=text_col), title_font=dict(color=text_col), zerolinecolor=grid_col)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=grid_col, tickfont=dict(color=text_col), title_font=dict(color=text_col), zerolinecolor=grid_col)
    return fig

def create_wind_rose_figure(rose_df, title_text):
    dir_map = {k: v.split("(")[1].split(")")[0] for k, v in SEKTOR_DETIL_MAP.items()}
    df_clean = rose_df[~rose_df["Direction"].isin(["CALM", "VARIABLE"])].copy()
    if df_clean.empty: return None
    
    df_clean["Arah Mata Angin"] = df_clean["Direction"].map(dir_map)
    df_clean = df_clean.dropna(subset=["Arah Mata Angin"]) # Pengaman error pada pemetaan
    if df_clean.empty: return None
    
    speeds = ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-45", ">45"]
    avail_speeds = [s for s in speeds if s in df_clean.columns]
    if not avail_speeds: return None
    
    melt_rose = df_clean.melt(id_vars=["Arah Mata Angin"], value_vars=avail_speeds, var_name="Kecepatan (Knot)", value_name="Frekuensi (%)")
    agg_rose = melt_rose.groupby(["Arah Mata Angin", "Kecepatan (Knot)"])["Frekuensi (%)"].mean(numeric_only=True).reset_index()
    
    fig_polar = px.bar_polar(
        agg_rose, r="Frekuensi (%)", theta="Arah Mata Angin", color="Kecepatan (Knot)",
        color_discrete_sequence=["#001f3f", "#0074D9", "#2ECC40", "#FFDC00", "#FF851B", "#FF4136", "#F012BE", "#B10DC9", "#111111"] if not is_dark else ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899", "#14B8A6", "#F97316", "#FFFFFF"],
        category_orders={"Kecepatan (Knot)": speeds}, template=chart_template
    )
    fig_polar = apply_wmo_style(fig_polar, title_text, "", "")
    grid_col = "rgba(255,255,255,0.2)" if is_dark else "rgba(0,0,0,0.15)"
    text_col = "#FAFAFA" if is_dark else "#0F1116"
    legend_bg = "rgba(14,17,23,0.85)" if is_dark else "rgba(255, 255, 255, 0.9)"
    
    fig_polar.update_layout(
        font=dict(color=text_col),
        legend=dict(
            title=dict(text="<b>Kecepatan (Knot):</b>", font=dict(color=text_col)),
            font=dict(color=text_col),
            orientation="v", yanchor="top", y=1, xanchor="left", x=1.08,
            bgcolor=legend_bg, bordercolor=grid_col, borderwidth=1
        ),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            angularaxis=dict(
                direction="clockwise", rotation=90, categoryorder="array", 
                categoryarray=list(dir_map.values()),
                tickfont=dict(color=text_col, size=12, weight="bold"),
                gridcolor=grid_col, linecolor=grid_col
            ),
            radialaxis=dict(
                showgrid=True, gridcolor=grid_col, ticksuffix="%",
                tickfont=dict(color=text_col, size=10),
                linecolor=grid_col
            )
        )
    )
    return fig_polar

def render_icao_interpretation(title, content, highlight):
    bg_div = "#16202B" if is_dark else "#F8F9FA"
    text_div = "#E2E8F0" if is_dark else "#0F1116"
    title_div = "#60A5FA" if is_dark else "#0B3C5D"
    border_left = "#3B82F6" if is_dark else "#0074D9"
    border_div = "#334155" if is_dark else "#CBD5E1"
    warn_text = "#F87171" if is_dark else "#D9534F"
    dash_col = "#475569" if is_dark else "#CBD5E1"
    
    st.markdown(f"""
        <div style="background-color: {bg_div}; border: 1px solid {border_div}; border-left: 5px solid {border_left}; padding: 16px 20px; border-radius: 8px; margin-top: 15px; margin-bottom: 25px; box-shadow: 0 2px 5px rgba(0,0,0,0.15);">
            <div style="font-size: 14px; font-weight: 700; color: {title_div} !important; margin-bottom: 8px;">
                📋 INTERPRETASI KLIMATOLOGIS OPERASIONAL ({title.upper()})
            </div>
            <div style="font-size: 13.5px; color: {text_div} !important; line-height: 1.6;">
                {content}
            </div>
            <div style="margin-top: 10px; padding-top: 10px; border-top: 1px dashed {dash_col}; font-size: 13px; color: {warn_text} !important; font-weight: 600;">
                ⚠️ Dampak Penerbangan (ICAO Annex 3 / BMKG): {highlight}
            </div>
        </div>
    """, unsafe_allow_html=True)


# ==========================================
# 5. DISPLAY UTAMA & INTERPRETASI DINAMIS
# ==========================================

# 5.A. RENDER BERANDA UTAMA (HOME)
if selected_param == "Home":
    st.markdown("""
        <div style="background: linear-gradient(135deg, #021B38 0%, #08335E 50%, #17548A 100%); 
                    padding: 45px 30px; 
                    border-radius: 12px; 
                    color: white; 
                    text-align: center; 
                    box-shadow: 0 8px 16px rgba(0,0,0,0.4); 
                    border-bottom: 5px solid #FFD700;
                    margin-bottom: 30px;">
            <h1 style="margin: 0; font-size: 38px; font-weight: 800; text-transform: uppercase; letter-spacing: 1.5px; color: #FFFFFF !important;">Sistem Informasi Meteorologi Penerbangan</h1>
            <p style="margin: 12px 0 20px 0; font-size: 18px; font-weight: 300; letter-spacing: 0.5px; opacity: 0.95; color: #FFFFFF !important;">
                 Analisis Data Meteorologi Taktis & Dukungan Keselamatan Operasi Udara
            </p>
            <hr style="border-top: 1px solid rgba(255,255,255,0.2); width: 60%; margin: auto;">
            <p style="margin-top: 25px; font-size: 16px; font-weight: 400; font-style: italic; color: #F1C40F !important;">
                "Swa Bhuwana Paksa"
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🌡️ **Analisis Termal & Visibilitas**\n\nMenyajikan pemetaan suhu, jarak pandang, dan tinggi dasar awan untuk penentuan status VFR/IFR.")
    with col2:
        st.success("🧭 **Wind Rose & Monsun**\n\nSpektrum arah dan kecepatan angin historis untuk rekomendasi operasional Landas Pacu (Runway).")
    with col3:
        st.warning("⚠️ **Peringatan Dini Operasional**\n\nInterpretasi ICAO terintegrasi guna mendeteksi ancaman cuaca terhadap keselamatan kru dan armada.")
    
    st.markdown("---")
    st.markdown(f"**Silakan pilih menu di panel sebelah kiri untuk memulai analisis data meteorologi.**")

# 5.B. RENDER DASHBOARD (PARAMETER LAINNYA)
else:
    st.markdown(f"### 📊 Dashboard Analisis: {param_options[selected_param].split('. ')[1]}")
    st.markdown("---")

    if selected_param in ["TempMaxMin", "RH"]:
        df_filtered = filter_df(data[selected_param])
        if df_filtered.empty:
            st.warning("⚠️ Data kosong pada filter terpilih. Silakan sesuaikan parameter bulan atau tahun di sidebar.")
        else:
            y_lbl = "Suhu Udara (°C)" if selected_param == "TempMaxMin" else "Kelembapan Relatif (%)"
            cols = ["00", "03", "06", "09", "12", "15", "18", "21", "Daily_Mean", "Max", "Min"]
            avail_cols = [c for c in cols if c in df_filtered.columns]
            agg_df = df_filtered.groupby("Tanggal")[avail_cols].mean(numeric_only=True).reset_index().sort_values("Tanggal")
            melted = agg_df.melt(id_vars="Tanggal", value_vars=avail_cols, var_name="Jam / Indikator", value_name="Nilai")
            
            fig_line = px.line(melted, x="Tanggal", y="Nilai", color="Jam / Indikator", markers=True, color_discrete_sequence=PALET_KATEGORI)
            fig_line.update_traces(line=dict(width=2), marker=dict(size=6), hovertemplate="<b>Tanggal %{x}</b><br>Nilai: <b>%{y:.1f}</b><extra></extra>")
            fig_line = apply_wmo_style(fig_line, f"Trend Harian Real-Time - {month_choice} ({selected_year})", "Tanggal Pengamatan", y_lbl)
            
            # Pengaman batas maksimum tanggal agar presisi
            max_tgl = int(agg_df["Tanggal"].max()) if not agg_df["Tanggal"].empty and pd.notna(agg_df["Tanggal"].max()) else 31
            fig_line.update_layout(xaxis=dict(tickmode="linear", dtick=1, range=[0.5, max_tgl + 0.5]))
            
            st.plotly_chart(fig_line, width="stretch", theme=None)
            
            try:
                if selected_param == "TempMaxMin" and "Max" in agg_df and "Min" in agg_df:
                    avg_max, avg_min = agg_df["Max"].mean(numeric_only=True), agg_df["Min"].mean(numeric_only=True)
                    ext_max = agg_df["Max"].max(numeric_only=True)
                    if pd.notna(avg_max) and pd.notna(ext_max):
                        render_icao_interpretation(
                            "Suhu Udara Synoptic",
                            f"Rata-rata suhu maksimum mencapai <b>{avg_max:.1f}°C</b> dengan ekstrem tertinggi <b>{ext_max:.1f}°C</b>, sementara rata-rata minimum berada pada <b>{avg_min:.1f}°C</b>. Rentang diurnal ini menunjukkan karakteristik pembentukan lapisan batas atmosfer yang aktif pada siang hari.",
                            f"Suhu ekstrem {ext_max:.1f}°C meningkatkan <i>Density Altitude</i> (penurunan kepadatan udara), yang secara langsung memperpanjang jarak lepas landas (<i>take-off run</i>) dan mengurangi daya angkat (<i>lift</i>) pesawat."
                        )
                elif selected_param == "RH":
                    avg_rh = agg_df["Daily_Mean"].mean(numeric_only=True) if "Daily_Mean" in agg_df else agg_df.mean(numeric_only=True).mean()
                    if pd.notna(avg_rh):
                        render_icao_interpretation(
                            "Kelembapan Relatif (RH)",
                            f"Kelembapan relatif rata-rata harian berada pada level <b>{avg_rh:.1f}%</b>. Pola saturasi tertinggi konsisten terjadi pada pengamatan dini hari (21.00 - 00.00 UTC / 04.00 - 07.00 WIB) akibat pendinginan radiatif permukaan.",
                            "Kelembapan tinggi (>85%) pada dini/pagi hari memicu probabilitas tinggi pembentukan kabut radiasi (<i>radiation fog</i>) dan embun, yang berpotensi menurunkan <i>Runway Visual Range</i> (RVR) di bawah batas minimum VFR."
                        )
            except Exception: 
                pass

    elif selected_param in ["TempFreq", "Vis", "HS"]:
        df_filtered = filter_df(data[selected_param])
        if df_filtered.empty:
            st.warning("⚠️ Data tidak ditemukan untuk filter ini. Silakan sesuaikan parameter bulan atau tahun di sidebar.")
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
            # Validasi & filter ketat untuk memastikan jam UTC berada eksklusif di rentang 0 - 23
            agg_v = df_filtered[df_filtered["Jam"] <= 23].groupby("Jam")[avail_cols].mean(numeric_only=True).reset_index().sort_values("Jam")
            hm_df = agg_v.melt(id_vars="Jam", value_vars=avail_cols, var_name="Kategori Batas", value_name="Persentase")
            
            fig_freq = px.line(hm_df, x="Jam", y="Persentase", color="Kategori Batas", markers=True, color_discrete_sequence=PALET_KATEGORI)
            fig_freq.update_traces(line=dict(width=2.5), marker=dict(size=7), hovertemplate="<b>Jam %{x:02d} UTC</b><br>Persentase: <b>%{y:.2f}%</b><extra></extra>")
            fig_freq = apply_wmo_style(fig_freq, f"Pola Distribusi Per Jam Observasi (UTC) - {month_choice}", "Jam Synoptic (UTC)", y_lbl)
            
            # SOLUSI INTI: Penguncian sumbu X secara presisi untuk mengeliminasi munculnya angka jam 24
            fig_freq.update_layout(
                xaxis=dict(
                    tickmode="array",
                    tickvals=[0, 3, 6, 9, 12, 15, 18, 21],
                    ticktext=["00", "03", "06", "09", "12", "15", "18", "21"],
                    range=[-0.5, 23.5],
                    zeroline=False
                )
            )
            
            st.plotly_chart(fig_freq, width="stretch", theme=None)
            
            try:
                mean_series = hm_df.groupby("Kategori Batas")["Persentase"].mean().dropna()
                if not mean_series.empty:
                    dom_cat = mean_series.idxmax()
                    dom_val = mean_series.max()
                    if selected_param == "TempFreq":
                        render_icao_interpretation(
                            "Distribusi Frekuensi Suhu",
                            f"Distribusi termal paling dominan berada pada rentang suhu <b>{dom_cat} °C</b> dengan frekuensi kemunculan rata-rata <b>{dom_val:.1f}%</b> di seluruh jam pengamatan synoptic.",
                            "Konsentrasi frekuensi pada suhu >30°C menuntut kewaspadaan terhadap <i>heat stress</i> pada kru <i>flightline</i> serta penurunan daya dorong (<i>thrust</i>) mesin turbin pesawat."
                        )
                    elif selected_param == "Vis":
                        vis_low = hm_df[hm_df["Kategori Batas"].isin(["<200", "<400", "<600", "<800", "<1500"])]["Persentase"].mean()
                        render_icao_interpretation(
                            "Jarak Pandang / Visibility",
                            f"Kondisi jarak pandang didominasi oleh kategori <b>{dom_cat} meter</b> ({dom_val:.1f}%). Akumulasi frekuensi jarak pandang buruk (<1500m) tercatat sebesar <b>{vis_low:.2f}%</b>, umumnya terjadi pada fase transisi subuh-pagi.",
                            "Jarak pandang di bawah 1500 meter merupakan batas kritis ICAO yang mewajibkan pemberlakuan prosedur penerbangan instrumen (IFR) atau penundaan operasi VFR taktis."
                        )
                    else:
                        render_icao_interpretation(
                            "Tinggi Dasar Awan / HS",
                            f"Frekuensi kejadian dasar awan terendah didominasi oleh rentang <b>{dom_cat} FT</b> dengan rata-rata probabilitas <b>{dom_val:.1f}%</b>.",
                            "Tinggi dasar awan di bawah 1500 FT (khususnya <1000 FT) masuk dalam parameter <i>Instrument Meteorological Conditions</i> (IMC), yang membutuhkan kesiapan sistem pemandu pendaratan instrumen (ILS/VOR)."
                        )
            except Exception: 
                pass

    elif selected_param == "Wind":
        df_w = filter_df(data["Wind"])
        if df_w.empty:
            st.warning("⚠️ Berkas Wind Rose kosong atau format kolom tidak sesuai.")
        else:
            tab_rose, tab_musim = st.tabs(["🧭 1. Spektrum Wind Rose (Umum)", "🌦️ 2. Integrasi Analisis Monsun & Sektor WMO 3600"])
            
            with tab_rose:
                fig_rose = create_wind_rose_figure(df_w, f"Mawar Angin Standar Penerbangan - {month_choice} ({selected_year})")
                if fig_rose: 
                    st.plotly_chart(fig_rose, width="stretch", theme=None)
                else: 
                    st.info("💡 Tidak ada komponen angin terdeteksi selain data CALM/VARIABLE.")
                
            with tab_musim:
                with st.expander("📖 PANDUAN REFERENSI DE-KODING SEKTOR ARAH ANGIN WMO CODE TABLE 3600"):
                    st.markdown("Sistem menerjemahkan pembagian **360° lingkaran kompas** menjadi **12 Sektor Utama Internasional** (lebar 30°/sektor) untuk validasi arah datang angin terhadap orientasi landas pacu:")
                    html_table = "<table class='wmo-table'><tr><th>Kode Sektor</th><th>Mata Angin Internasional</th><th>Arti Pengamatan</th><th>Arah Sudut Kompas (Derajat)</th></tr>"
                    for code, details in SEKTOR_DETIL_MAP.items():
                        parts = details.split("|")
                        lbl, desc, deg = parts[0].split("(")[1].replace(")", "").strip(), parts[0].split(")")[1].strip(), parts[1].strip()
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
                        if fig_rm: 
                            st.plotly_chart(fig_rm, width="stretch", theme=None)
                        
                    with col_b:
                        df_g_musim = df_musim[~df_musim["Direction"].isin(["CALM", "VARIABLE"])].groupby(["Direction", "Bulan", "Bulan_Idx"])["Total"].mean(numeric_only=True).reset_index().sort_values("Bulan_Idx")
                        df_g_musim["Sektor_Deskriptif"] = df_g_musim["Direction"].map(SEKTOR_DETIL_MAP).fillna(df_g_musim["Direction"])
                        
                        fig_m_bar = px.bar(
                            df_g_musim, x="Sektor_Deskriptif", y="Total", color="Bulan",
                            barmode="group", color_discrete_sequence=PALET_MUSIM_BAR,
                            category_orders={"Sektor_Deskriptif": list(SEKTOR_DETIL_MAP.values()), "Bulan": bulan_musim}
                        )
                        fig_m_bar = apply_wmo_style(fig_m_bar, f"Variabilitas Bulanan Sektor WMO 3600 ({pilihan_musim.split(' ')[0]})", "Sektor Arah Angin & Sudut Kompas WMO 3600", "Persentase Kejadian / Frekuensi (%)")
                        fig_m_bar.update_xaxes(type="category", categoryorder="array", categoryarray=list(SEKTOR_DETIL_MAP.values()))
                        
                        st.plotly_chart(fig_m_bar, width="stretch", theme=None)
                        
                    c_val = df_musim[df_musim["Direction"] == "CALM"]["Total"].mean()
                    try:
                        mean_wind_series = df_g_musim.groupby("Sektor_Deskriptif")["Total"].mean().dropna()
                        if not mean_wind_series.empty:
                            dom_wind_dir = mean_wind_series.idxmax()
                            dom_wind_val = mean_wind_series.max()
                            c_val_str = f"{c_val:.2f}%" if pd.notna(c_val) else "0.00%"
                            render_icao_interpretation(
                                f"Sirkulasi Angin Musim {pilihan_musim.split(' ')[0]}",
                                f"Arah angin dominan bertiup dari sektor <b>{dom_wind_dir}</b> dengan probabilitas <b>{dom_wind_val:.1f}%</b>. Kondisi angin tenang (<i>Calm Wind</i> < 1 Knot) tercatat sebesar <b>{c_val_str}</b>, menunjukkan variabilitas termal lokal yang signifikan pada pagi hari.",
                                "Sektor dominan ini menentukan penentuan arah lepas landas dan pendaratan (<i>Runway in Use</i>) untuk meminimalkan komponen angin silang (<i>Crosswind</i>) atau angin ekor (<i>Tailwind</i>) melebihi limitasi 15 knot sesuai ICAO Annex 14."
                            )
                    except Exception: 
                        pass
                else:
                    st.warning("⚠️ Data untuk sirkulasi musim ini tidak ditemukan pada berkas Excel Anda.")
