import re
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera, durbin_watson

# ============================================================
# KONFIGURASI
# ============================================================
st.set_page_config(
    page_title="Fiscal & Regional Development Analytics",
    page_icon="📊",
    layout="wide",
)

DATA_URL = (
    "https://raw.githubusercontent.com/Alam-A99/DATAAKSI/"
    "refs/heads/main/dataset_fiskal_97_hipoFull.csv"
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: rgba(120, 120, 120, 0.08);
        border: 1px solid rgba(120, 120, 120, 0.18);
        padding: 14px 16px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# FUNGSI DATA
# ============================================================
@st.cache_data(ttl=3600, show_spinner="Mengambil dataset dari GitHub...")
def load_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def find_column(columns, candidates):
    """Cari kolom dengan pencocokan nama persis/parsial secara toleran."""
    normalized = {normalize_name(c): c for c in columns}
    for candidate in candidates:
        key = normalize_name(candidate)
        if key in normalized:
            return normalized[key]
    for candidate in candidates:
        key = normalize_name(candidate)
        if len(key) >= 3:
            for norm, original in normalized.items():
                if key in norm or norm in key:
                    return original
    return None


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Konversi kolom numerik berbasis isi; jangan mengubah kolom kategori.
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            cleaned = (
                df[col].astype(str).str.strip()
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
            )
            converted = pd.to_numeric(cleaned, errors="coerce")
            non_empty = df[col].notna().sum()
            if non_empty > 0 and converted.notna().sum() / non_empty >= 0.85:
                df[col] = converted
    return df


def to_excel_bytes(df: pd.DataFrame):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")
    return buffer.getvalue()


# ============================================================
# SIDEBAR + LOAD
# ============================================================
st.title("📊 Fiscal & Regional Development Analytics")
st.caption("Dashboard eksplorasi data fiskal dan pembangunan daerah")
with st.sidebar:
    st.header("Pengaturan")
    st.write("Sumber data: GitHub CSV")
    if st.button("🔄 Muat ulang dataset", use_container_width=True):
        load_data.clear()
        st.rerun()

try:
    raw_df = load_data(DATA_URL)
    df = prepare_data(raw_df)
except Exception as e:
    st.error("Dataset gagal dibaca. Pastikan URL GitHub bersifat publik dan file CSV dapat diakses.")
    st.exception(e)
    st.stop()

if df.empty:
    st.warning("Dataset berhasil dibaca, tetapi tidak memiliki baris data.")
    st.stop()

# Deteksi kolom identitas umum
region_col = find_column(
    df.columns,
    ["Kota", "Kabupaten/Kota", "Kabupaten_Kota", "Nama Daerah", "Daerah", "Wilayah", "Kabupaten", "city", "region"]
)
year_col = find_column(df.columns, ["Tahun", "Year", "TA", "Periode"])

# Filter interaktif
with st.sidebar:
    st.subheader("Filter data")
    filtered = df.copy()

    if region_col:
        regions = sorted(filtered[region_col].dropna().astype(str).unique().tolist())
        selected_regions = st.multiselect(
            f"Pilih {region_col}",
            regions,
            default=regions[:min(5, len(regions))],
        )
        if selected_regions:
            filtered = filtered[filtered[region_col].astype(str).isin(selected_regions)]
        else:
            filtered = filtered.iloc[0:0]

    if year_col:
        years_numeric = pd.to_numeric(filtered[year_col], errors="coerce")
        unique_years = sorted(years_numeric.dropna().unique().tolist())
        if unique_years:
            year_range = st.slider(
                f"Rentang {year_col}",
                min_value=int(min(unique_years)),
                max_value=int(max(unique_years)),
                value=(int(min(unique_years)), int(max(unique_years))),
            )
            filtered = filtered[
                pd.to_numeric(filtered[year_col], errors="coerce").between(year_range[0], year_range[1])
            ]

numeric_cols = filtered.select_dtypes(include=np.number).columns.tolist()
categorical_cols = filtered.select_dtypes(exclude=np.number).columns.tolist()

# KPI
k1, k2, k3, k4 = st.columns(4)
k1.metric("Observasi", f"{len(filtered):,}")
k2.metric("Variabel", f"{filtered.shape[1]:,}")
k3.metric("Wilayah unik", f"{filtered[region_col].nunique():,}" if region_col and not filtered.empty else "Tidak terdeteksi")
k4.metric("Tahun/periode", f"{filtered[year_col].nunique():,}" if year_col and not filtered.empty else "Tidak terdeteksi")

if filtered.empty:
    st.warning("Filter tidak menghasilkan observasi. Pilih minimal satu wilayah dan periksa rentang tahun.")
    st.stop()

tabs = st.tabs([
    "Overview",
    "Fiscal Analytics",
    "Regional Comparison",
    "Visualization",
    "Correlation",
    "OLS Regression",
    "Diagnostics",
    "Data & Export",
])

# ============================================================
# TAB 1: OVERVIEW
# ============================================================
with tabs[0]:
    st.subheader("Ringkasan dataset")
    left, right = st.columns([1.2, 1])
    with left:
        st.write("**Cuplikan data**")
        st.dataframe(filtered.head(20), use_container_width=True)
    with right:
        st.write("**Kualitas data**")
        quality = pd.DataFrame({
            "Kolom": filtered.columns,
            "Tipe data": [str(filtered[c].dtype) for c in filtered.columns],
            "Missing": [int(filtered[c].isna().sum()) for c in filtered.columns],
            "Missing (%)": [round(filtered[c].isna().mean() * 100, 2) for c in filtered.columns],
            "Nilai unik": [int(filtered[c].nunique(dropna=True)) for c in filtered.columns],
        })
        st.dataframe(quality, use_container_width=True, hide_index=True)
    if numeric_cols:
        st.write("**Statistik deskriptif variabel numerik**")
        st.dataframe(filtered[numeric_cols].describe().T, use_container_width=True)

# ============================================================
# TAB 2: FISCAL ANALYTICS
# ============================================================
with tabs[1]:
    st.subheader("Analisis fiskal")
    fiscal_keywords = [
        "pad", "pendapatan", "transfer", "belanja", "pajak", "retribusi",
        "dau", "dak", "dbhp", "dbH", "silpa", "apbd", "fiskal", "penerimaan",
        "bm_pendidikan", "bm_kesehatan", "bm_infrastruktur"
    ]
    suggested = [
        c for c in numeric_cols
        if any(normalize_name(k) in normalize_name(c) for k in fiscal_keywords)
    ]
    default_metric = suggested[0] if suggested else (numeric_cols[0] if numeric_cols else None)
    if numeric_cols:
        metric = st.selectbox(
            "Pilih indikator fiskal/numerik",
            numeric_cols,
            index=numeric_cols.index(default_metric),
        )
        a, b = st.columns(2)
        with a:
            st.metric(f"Rata-rata {metric}", f"{filtered[metric].mean():,.3f}" if filtered[metric].notna().any() else "NA")
            fig = px.histogram(filtered, x=metric, nbins=30, title=f"Distribusi {metric}")
            st.plotly_chart(fig, use_container_width=True)
        with b:
            st.metric(f"Median {metric}", f"{filtered[metric].median():,.3f}" if filtered[metric].notna().any() else "NA")
            if region_col:
                agg = filtered.groupby(region_col, dropna=False)[metric].mean().reset_index().sort_values(metric, ascending=False)
                fig = px.bar(agg, x=region_col, y=metric, title=f"Rata-rata {metric} per wilayah")
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Kolom wilayah tidak terdeteksi; grafik per wilayah dilewati.")
    else:
        st.warning("Tidak ditemukan kolom numerik. Periksa format data CSV.")

# ============================================================
# TAB 3: REGIONAL COMPARISON
# ============================================================
with tabs[2]:
    st.subheader("Perbandingan wilayah")
    if region_col and numeric_cols:
        comp_metric = st.selectbox("Indikator untuk dibandingkan", numeric_cols, key="comp_metric")
        agg = filtered.groupby(region_col, dropna=False)[comp_metric].agg(["mean", "median", "sum", "count"]).reset_index()
        agg.columns = [region_col, "Rata-rata", "Median", "Total", "Jumlah observasi"]
        st.dataframe(agg.sort_values("Rata-rata", ascending=False), use_container_width=True, hide_index=True)
        fig = px.bar(agg.sort_values("Rata-rata", ascending=False), x=region_col, y="Rata-rata",
                     hover_data=["Median", "Total", "Jumlah observasi"],
                     title=f"Perbandingan rata-rata {comp_metric}")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Perbandingan wilayah membutuhkan kolom wilayah dan setidaknya satu kolom numerik.")

# ============================================================
# TAB 4: VISUALIZATION
# ============================================================
with tabs[3]:
    st.subheader("Visualisasi interaktif")
    if numeric_cols:
        chart_type = st.radio("Jenis grafik", ["Tren garis", "Scatter plot", "Box plot"], horizontal=True)
        x_options = df.columns.tolist()
        if chart_type == "Tren garis":
            x_default = year_col if year_col else (categorical_cols[0] if categorical_cols else df.columns[0])
            x_col = st.selectbox("Sumbu X", x_options, index=x_options.index(x_default), key="line_x")
            y_col = st.selectbox("Sumbu Y", numeric_cols, key="line_y")
            plot_df = filtered.groupby(x_col, dropna=False)[y_col].mean().reset_index() if x_col in categorical_cols or x_col == year_col else filtered
            fig = px.line(plot_df, x=x_col, y=y_col, markers=True, title=f"{y_col} menurut {x_col}")
        elif chart_type == "Scatter plot":
            x_col = st.selectbox("Variabel X", numeric_cols, key="scatter_x")
            y_col = st.selectbox("Variabel Y", [c for c in numeric_cols if c != x_col] or numeric_cols, key="scatter_y")
            color_col = st.selectbox("Warna (opsional)", ["Tidak ada"] + categorical_cols, key="scatter_color")
            fig = px.scatter(filtered, x=x_col, y=y_col,
                             color=None if color_col == "Tidak ada" else color_col,
                             hover_name=region_col if region_col else None,
                             title=f"Hubungan {x_col} dan {y_col}")
        else:
            y_col = st.selectbox("Variabel untuk box plot", numeric_cols, key="box_y")
            color_col = st.selectbox("Kelompok (opsional)", ["Tidak ada"] + categorical_cols, key="box_color")
            fig = px.box(filtered, y=y_col, x=None if color_col == "Tidak ada" else color_col,
                         title=f"Distribusi {y_col}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Tidak ada kolom numerik untuk divisualisasikan.")

# ============================================================
# TAB 5: CORRELATION
# ============================================================
with tabs[4]:
    st.subheader("Matriks korelasi")
    if len(numeric_cols) >= 2:
        corr_cols = st.multiselect("Pilih variabel numerik", numeric_cols, default=numeric_cols[:min(12, len(numeric_cols))])
        if len(corr_cols) >= 2:
            corr = filtered[corr_cols].corr(numeric_only=True)
            fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1, title="Korelasi Pearson")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Korelasi menunjukkan asosiasi linear, bukan bukti hubungan sebab-akibat.")
        else:
            st.info("Pilih setidaknya dua variabel.")
    else:
        st.info("Korelasi membutuhkan minimal dua kolom numerik.")

# ============================================================
# TAB 6: OLS REGRESSION
# ============================================================
with tabs[5]:
    st.subheader("Regresi linear OLS")
    st.caption("Gunakan variabel yang relevan secara teori. Hasil regresi tidak otomatis menunjukkan hubungan kausal.")
    if len(numeric_cols) >= 2:
        y_var = st.selectbox("Variabel dependen (Y)", numeric_cols, key="ols_y")
        x_candidates = [c for c in numeric_cols if c != y_var]
        x_vars = st.multiselect("Variabel independen (X)", x_candidates, default=x_candidates[:min(3, len(x_candidates))], key="ols_x")
        if st.button("Estimasi model OLS", type="primary"):
            if not x_vars:
                st.error("Pilih minimal satu variabel independen.")
            else:
                model_df = filtered[[y_var] + x_vars].replace([np.inf, -np.inf], np.nan).dropna()
                if len(model_df) <= len(x_vars) + 1:
                    st.error("Observasi valid tidak cukup untuk mengestimasi model ini.")
                else:
                    X = sm.add_constant(model_df[x_vars], has_constant="add")
                    y = model_df[y_var]
                    try:
                        model = sm.OLS(y, X).fit()
                        st.write(f"Observasi model: **{int(model.nobs)}**")
                        st.text(model.summary().as_text())
                        coef = pd.DataFrame({
                            "Variabel": model.params.index,
                            "Koefisien": model.params.values,
                            "Std. Error": model.bse.values,
                            "p-value": model.pvalues.values,
                            "CI bawah": model.conf_int()[0].values,
                            "CI atas": model.conf_int()[1].values,
                        })
                        st.dataframe(coef, use_container_width=True, hide_index=True)
                        pred_df = pd.DataFrame({"Aktual": y, "Prediksi": model.predict(X)})
                        fig = px.scatter(pred_df, x="Aktual", y="Prediksi", title="Nilai aktual vs prediksi")
                        lo = min(pred_df.min().min(), pred_df.max().min())
                        hi = max(pred_df.max().max(), pred_df.min().max())
                        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Garis 45°"))
                        st.plotly_chart(fig, use_container_width=True)
                        st.session_state["ols_model_summary"] = model.summary().as_text()
                        st.session_state["ols_coef"] = coef
                        st.session_state["ols_resid"] = model.resid
                        st.session_state["ols_fitted"] = model.fittedvalues
                    except Exception as e:
                        st.error(f"Model gagal diestimasi: {e}")
    else:
        st.info("Regresi OLS membutuhkan minimal dua variabel numerik.")

# ============================================================
# TAB 7: DIAGNOSTICS
# ============================================================
with tabs[6]:
    st.subheader("Diagnostik model")
    if "ols_resid" not in st.session_state:
        st.info("Estimasi model pada tab OLS Regression terlebih dahulu.")
    else:
        resid = pd.Series(st.session_state["ols_resid"]).dropna()
        fitted = pd.Series(st.session_state["ols_fitted"]).loc[resid.index]
        d1, d2, d3 = st.columns(3)
        jb_stat, jb_p, skew, kurtosis = jarque_bera(resid)
        d1.metric("Jarque–Bera p-value", f"{jb_p:.4f}")
        d2.metric("Durbin–Watson", f"{durbin_watson(resid):.4f}")
        fig = px.scatter(x=fitted, y=resid, labels={"x": "Fitted values", "y": "Residual"},
                         title="Residual vs Fitted")
        fig.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)
        st.write("**Interpretasi ringkas**")
        st.write("- Jarque–Bera: p-value kecil dapat mengindikasikan residual tidak normal.")
        st.write("- Durbin–Watson mendekati 2 umumnya menunjukkan autokorelasi orde pertama yang lebih rendah; interpretasi bergantung pada struktur data.")
        st.write("- Residual vs fitted membantu memeriksa pola varians dan bentuk hubungan.")
        st.warning("Untuk Breusch–Pagan dan VIF, pilih kembali variabel model dan jalankan estimasi; interpretasi perlu mempertimbangkan jenis data dan spesifikasi model.")

# ============================================================
# TAB 8: DATA & EXPORT
# ============================================================
with tabs[7]:
    st.subheader("Data terfilter")
    st.write(f"Menampilkan **{len(filtered):,} baris** dari {len(df):,} baris awal.")
    st.dataframe(filtered, use_container_width=True, height=480)
    csv_data = filtered.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ Unduh CSV", data=csv_data, file_name="dataset_fiskal_terfilter.csv", mime="text/csv")
    try:
        st.download_button(
            "⬇️ Unduh Excel",
            data=to_excel_bytes(filtered),
            file_name="dataset_fiskal_terfilter.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as e:
        st.warning(f"Ekspor Excel belum tersedia: {e}")

with st.expander("Nama kolom yang terdeteksi"):
    st.write("**Kolom wilayah:**", region_col or "Tidak terdeteksi otomatis")
    st.write("**Kolom tahun/periode:**", year_col or "Tidak terdeteksi otomatis")
    st.write("**Semua kolom:**")
    st.code(", ".join(map(str, df.columns)))
