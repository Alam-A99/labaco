import re
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =====================================================
# KONFIGURASI
# =====================================================
st.set_page_config(
    page_title="Dashboard Perbandingan Kota",
    page_icon="📊",
    layout="wide",
)

DATA_URL = (
    "https://raw.githubusercontent.com/Alam-A99/DATAAKSI/"
    "refs/heads/main/dataset_fiskal_97_hipoFull.csv"
)

st.title("📊 Dashboard Fiskal dan Pembangunan Daerah")
st.caption("Eksplorasi data dan perbandingan kota/kabupaten secara interaktif")

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data(ttl=3600)
def load_data(url):
    data = pd.read_csv(url)
    data.columns = [str(c).strip() for c in data.columns]
    return data

def norm(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())

def detect_col(columns, candidates):
    norm_map = {norm(c): c for c in columns}
    for candidate in candidates:
        key = norm(candidate)
        if key in norm_map:
            return norm_map[key]
    for candidate in candidates:
        key = norm(candidate)
        if len(key) >= 3:
            for normalized, original in norm_map.items():
                if key in normalized:
                    return original
    return None

try:
    df = load_data(DATA_URL)
except Exception as e:
    st.error("Dataset tidak berhasil dimuat. Pastikan URL GitHub benar dan file bersifat publik.")
    st.exception(e)
    st.stop()

if df.empty:
    st.warning("Dataset tidak memiliki baris data.")
    st.stop()

# Ubah kolom menjadi numerik jika sebagian besar nilainya memang angka
for col in df.columns:
    if df[col].dtype == "object":
        candidate = pd.to_numeric(
            df[col].astype(str).str.strip().str.replace(",", "", regex=False),
            errors="coerce",
        )
        non_missing = df[col].notna().sum()
        if non_missing and candidate.notna().sum() / non_missing >= 0.85:
            df[col] = candidate

region_col = detect_col(
    df.columns,
    ["Kota", "Kabupaten/Kota", "Kabupaten_Kota", "Nama Daerah", "Daerah", "Wilayah", "Kabupaten", "City", "Region"],
)
year_col = detect_col(df.columns, ["Tahun", "Year", "Periode", "TA"])

num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

if not region_col:
    st.warning("Kolom kota/kabupaten belum terdeteksi otomatis. Kolom tersedia: " + ", ".join(df.columns))
    region_col = st.selectbox("Pilih kolom kota/kabupaten", df.columns)

# =====================================================
# FILTER SIDEBAR
# =====================================================
with st.sidebar:
    st.header("Filter Dashboard")
    if st.button("🔄 Muat ulang data", use_container_width=True):
        load_data.clear()
        st.rerun()

    region_values = sorted(df[region_col].dropna().astype(str).unique().tolist())
    selected_regions = st.multiselect(
        "Pilih kota/kabupaten",
        region_values,
        default=region_values[:min(5, len(region_values))],
    )

    view = df[df[region_col].astype(str).isin(selected_regions)].copy()

    if year_col:
        years = pd.to_numeric(view[year_col], errors="coerce").dropna()
        if not years.empty:
            low, high = int(years.min()), int(years.max())
            year_range = st.slider(
                "Rentang tahun",
                min_value=low,
                max_value=high,
                value=(low, high),
            )
            view = view[
                pd.to_numeric(view[year_col], errors="coerce").between(
                    year_range[0], year_range[1]
                )
            ]

if view.empty:
    st.warning("Pilih minimal satu kota/kabupaten pada sidebar.")
    st.stop()

# =====================================================
# KPI
# =====================================================
m1, m2, m3, m4 = st.columns(4)
m1.metric("Jumlah observasi", f"{len(view):,}")
m2.metric("Kota/kabupaten", f"{view[region_col].nunique():,}")
m3.metric("Jumlah variabel", f"{view.shape[1]:,}")
m4.metric(
    "Rentang tahun",
    f"{int(pd.to_numeric(view[year_col], errors='coerce').min())}–{int(pd.to_numeric(view[year_col], errors='coerce').max())}"
    if year_col and pd.to_numeric(view[year_col], errors="coerce").notna().any()
    else "Tidak tersedia",
)

# =====================================================
# TAB
# =====================================================
tab_overview, tab_fiscal, tab_compare, tab_development, tab_visual, tab_corr, tab_data = st.tabs([
    "Overview",
    "Fiscal Analytics",
    "Perbandingan Kota",
    "Indikator Pembangunan",
    "Visualization",
    "Correlation",
    "Data",
])

with tab_overview:
    st.subheader("Overview dataset")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.write("**Cuplikan data**")
        st.dataframe(view.head(15), use_container_width=True)
    with c2:
        st.write("**Ringkasan kualitas data**")
        quality = pd.DataFrame({
            "Kolom": view.columns,
            "Tipe": [str(view[c].dtype) for c in view.columns],
            "Missing": [int(view[c].isna().sum()) for c in view.columns],
        })
        st.dataframe(quality, use_container_width=True, hide_index=True)
    if num_cols:
        st.write("**Statistik deskriptif**")
        st.dataframe(view[num_cols].describe().T, use_container_width=True)

with tab_fiscal:
    st.subheader("Fiscal Analytics")
    fiscal_terms = [
        "pad", "transfer", "pendapatan", "belanja", "pajak", "retribusi",
        "dau", "dak", "dbhp", "apbd", "penerimaan", "bm"
    ]
    fiscal_cols = [
        c for c in num_cols
        if any(norm(term) in norm(c) for term in fiscal_terms)
    ]
    options = fiscal_cols if fiscal_cols else num_cols
    if options:
        selected_metric = st.selectbox("Pilih indikator fiskal", options, key="fiscal_metric")
        summary = view.groupby(region_col)[selected_metric].agg(["mean", "median", "sum"]).reset_index()
        summary.columns = [region_col, "Rata-rata", "Median", "Total"]
        st.dataframe(summary.sort_values("Rata-rata", ascending=False), use_container_width=True, hide_index=True)
        fig = px.bar(
            summary.sort_values("Rata-rata", ascending=False),
            x=region_col, y="Rata-rata",
            title=f"Perbandingan rata-rata {selected_metric} antarwilayah",
        )
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Tidak ada variabel numerik yang bisa dianalisis.")

with tab_compare:
    st.subheader("Perbandingan kota/kabupaten")
    if num_cols:
        metric = st.selectbox("Pilih indikator untuk dibandingkan", num_cols, key="compare_metric")
        compare_type = st.radio(
            "Ukuran perbandingan",
            ["Rata-rata", "Total", "Median"],
            horizontal=True,
        )
        agg_func = {"Rata-rata": "mean", "Total": "sum", "Median": "median"}[compare_type]
        comparison = view.groupby(region_col)[metric].agg(agg_func).reset_index(name=compare_type)
        comparison = comparison.sort_values(compare_type, ascending=False)
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        fig = px.bar(
            comparison,
            x=region_col, y=compare_type,
            title=f"{compare_type} {metric} menurut kota/kabupaten",
            hover_data={compare_type: ":,.3f"},
        )
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

        if year_col:
            st.markdown("**Tren indikator menurut kota/kabupaten**")
            trend = view.groupby([year_col, region_col], dropna=False)[metric].mean().reset_index()
            fig2 = px.line(
                trend, x=year_col, y=metric, color=region_col, markers=True,
                title=f"Tren {metric} menurut wilayah",
            )
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Tidak ada variabel numerik untuk dibandingkan.")

with tab_development:
    st.subheader("Indikator pembangunan daerah")
    development_terms = [
        "ipm", "kemiskinan", "pengangguran", "tpt", "pertumbuhan",
        "pdrb", "gini", "pendidikan", "kesehatan", "harapanhidup",
        "stunting", "infrastruktur", "ekonomi", "indeks"
    ]
    dev_cols = [
        c for c in num_cols
        if any(norm(term) in norm(c) for term in development_terms)
    ]
    options = dev_cols if dev_cols else num_cols
    if options:
        dev_metric = st.selectbox("Pilih indikator pembangunan", options, key="development_metric")
        dev_summary = view.groupby(region_col)[dev_metric].mean().reset_index()
        dev_summary = dev_summary.sort_values(dev_metric, ascending=False)
        fig = px.bar(
            dev_summary, x=region_col, y=dev_metric,
            title=f"Rata-rata {dev_metric} menurut wilayah",
        )
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(dev_summary, use_container_width=True, hide_index=True)
    else:
        st.info("Tidak ditemukan indikator numerik.")

with tab_visual:
    st.subheader("Visualization")
    if num_cols:
        chart_type = st.selectbox(
            "Jenis grafik",
            ["Tren garis", "Scatter plot", "Histogram", "Box plot"],
        )
        if chart_type == "Tren garis":
            x_col = year_col or st.selectbox("Sumbu X", view.columns, key="trend_x")
            y_col = st.selectbox("Indikator (Y)", num_cols, key="trend_y")
            trend = view.groupby([x_col, region_col], dropna=False)[y_col].mean().reset_index()
            fig = px.line(trend, x=x_col, y=y_col, color=region_col, markers=True)
        elif chart_type == "Scatter plot":
            x_col = st.selectbox("Variabel X", num_cols, key="scatter_x")
            y_choices = [c for c in num_cols if c != x_col] or num_cols
            y_col = st.selectbox("Variabel Y", y_choices, key="scatter_y")
            fig = px.scatter(view, x=x_col, y=y_col, color=region_col, hover_data=[year_col] if year_col else None)
        elif chart_type == "Histogram":
            x_col = st.selectbox("Variabel", num_cols, key="hist_x")
            fig = px.histogram(view, x=x_col, color=region_col, barmode="overlay", opacity=0.65)
        else:
            y_col = st.selectbox("Variabel", num_cols, key="box_y")
            fig = px.box(view, x=region_col, y=y_col, points="outliers")
            fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Tidak ada kolom numerik untuk grafik.")

with tab_corr:
    st.subheader("Correlation Analysis")
    if len(num_cols) >= 2:
        corr_options = st.multiselect(
            "Pilih variabel numerik",
            num_cols,
            default=num_cols[:min(10, len(num_cols))],
        )
        if len(corr_options) >= 2:
            corr = view[corr_options].corr()
            fig = px.imshow(
                corr, text_auto=".2f", aspect="auto",
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                title="Matriks korelasi Pearson",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Korelasi mengukur hubungan linear, bukan membuktikan sebab-akibat.")
        else:
            st.info("Pilih setidaknya dua variabel.")
    else:
        st.info("Diperlukan setidaknya dua variabel numerik.")

with tab_data:
    st.subheader("Data dan unduhan")
    st.dataframe(view, use_container_width=True, height=450)
    st.download_button(
        "⬇️ Unduh data terfilter (CSV)",
        data=view.to_csv(index=False).encode("utf-8-sig"),
        file_name="data_fiskal_perbandingan_kota.csv",
        mime="text/csv",
    )

with st.expander("Informasi kolom"):
    st.write("Kolom wilayah:", region_col)
    st.write("Kolom tahun:", year_col or "Tidak terdeteksi")
    st.write("Kolom numerik:", num_cols)
