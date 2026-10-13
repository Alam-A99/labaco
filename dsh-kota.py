
# ============================================================
# FISCAL & REGIONAL DEVELOPMENT ANALYTICS DASHBOARD
# Dataset: DATAAKSI - dataset_fiskal_97_hipoFull.csv
# Main file: app.py
# ============================================================

import io
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

from statsmodels.stats.diagnostic import (
    het_breuschpagan,
    het_white,
)
from statsmodels.stats.stattools import (
    jarque_bera,
    durbin_watson,
)

warnings.filterwarnings("ignore")

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fiscal Analytics | DATAAKSI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "Alam-A99/DATAAKSI/refs/heads/main/"
    "dataset_fiskal_97_hipoFull.csv"
)

# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .stApp {
        background-color: #F5F7FB;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        color: #17365D;
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #E2E8F0;
        padding: 18px;
        border-radius: 12px;
    }

    [data-testid="stSidebar"] {
        background-color: #EAF0F8;
    }

    div[data-testid="stPlotlyChart"] {
        background-color: white;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. LOAD DATA
# ============================================================

@st.cache_data(ttl=3600, show_spinner="Memuat dataset fiskal...")
def load_data(url):
    data = pd.read_csv(url)

    # Standardisasi nama kolom
    data.columns = [
        str(col).strip().replace("\ufeff", "")
        for col in data.columns
    ]

    # Bersihkan teks pada kolom kategori
    for col in data.select_dtypes(include=["object"]).columns:
        data[col] = data[col].apply(
            lambda x: x.strip() if isinstance(x, str) else x
        )

    # Konversi kolom numerik jika seluruh nilai valid
    for col in data.columns:
        if col not in ["Kota", "Tahun"]:
            converted = pd.to_numeric(data[col], errors="coerce")

            if converted.notna().sum() > 0:
                data[col] = converted

    if "Tahun" in data.columns:
        data["Tahun"] = pd.to_numeric(
            data["Tahun"], errors="coerce"
        )

    return data


try:
    df = load_data(DATA_URL)
except Exception as e:
    st.error(f"Gagal memuat dataset: {e}")
    st.info(
        "Periksa koneksi internet atau unggah file CSV "
        "melalui menu DATA pada sidebar."
    )

    uploaded = st.sidebar.file_uploader(
        "Unggah dataset CSV",
        type=["csv"],
    )

    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        st.stop()


# ============================================================
# 4. DATA PREPARATION
# ============================================================

if "Kota" not in df.columns or "Tahun" not in df.columns:
    st.error(
        "Dataset harus memiliki kolom identitas Kota dan Tahun."
    )
    st.stop()

df["Kota"] = df["Kota"].astype("string").str.strip()
df["Tahun"] = pd.to_numeric(df["Tahun"], errors="coerce")

numeric_cols = df.select_dtypes(
    include=np.number
).columns.tolist()

numeric_cols = [
    c for c in numeric_cols if c != "Tahun"
]

if not numeric_cols:
    st.error("Tidak ditemukan variabel numerik untuk dianalisis.")
    st.stop()

# Pengelompokan indikator secara otomatis
fiscal_keywords = [
    "PAD", "DBH", "DAU", "DAK", "TRANSFER",
    "PENDAPATAN", "FISKAL", "PAJAK", "PNBP"
]

spending_keywords = [
    "BM_", "BELANJA", "PENDIDIKAN", "KESEHATAN",
    "PERUMAHAN", "LINGKUNGAN", "EKONOMI",
    "INFRASTRUKTUR", "PARIWISATA",
    "PERLINDUNGAN_SOSIAL"
]

social_keywords = [
    "IPM", "KEMISKINAN", "PENGANGGURAN",
    "KETIMPANGAN", "GINI", "PENDIDIKAN",
    "KESEHATAN"
]

fiscal_cols = [
    c for c in numeric_cols
    if any(k in c.upper() for k in fiscal_keywords)
]

spending_cols = [
    c for c in numeric_cols
    if any(k in c.upper() for k in spending_keywords)
]

social_cols = [
    c for c in numeric_cols
    if any(k in c.upper() for k in social_keywords)
]

# Hindari duplikasi pilihan indikator
fiscal_cols = list(dict.fromkeys(fiscal_cols))
spending_cols = list(dict.fromkeys(spending_cols))
social_cols = list(dict.fromkeys(social_cols))


# ============================================================
# 5. SIDEBAR FILTERS
# ============================================================

st.sidebar.title("📊 FISCAL ANALYTICS")
st.sidebar.caption("DATAAKSI | Regional Development")

st.sidebar.markdown("---")
st.sidebar.subheader("Filter Data")

years = sorted(
    df["Tahun"].dropna().astype(int).unique().tolist()
)

cities = sorted(
    df["Kota"].dropna().astype(str).unique().tolist()
)

selected_years = st.sidebar.multiselect(
    "Pilih tahun",
    options=years,
    default=years,
)

selected_cities = st.sidebar.multiselect(
    "Pilih wilayah",
    options=cities,
    default=cities,
)

if not selected_years or not selected_cities:
    st.warning("Pilih minimal satu tahun dan satu wilayah.")
    st.stop()

filtered = df[
    df["Tahun"].isin(selected_years)
    & df["Kota"].isin(selected_cities)
].copy()

if filtered.empty:
    st.warning("Tidak ada observasi untuk filter tersebut.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Dataset: {len(df):,} observasi | "
    f"{len(numeric_cols)} indikator numerik"
)

if st.sidebar.button("🔄 Refresh dataset"):
    load_data.clear()
    st.rerun()


# ============================================================
# 6. HEADER
# ============================================================

st.title("Fiscal & Regional Development Analytics")
st.markdown(
    "### Dashboard Analisis Fiskal, Belanja Modal, "
    "dan Kinerja Pembangunan Daerah"
)

st.caption(
    "Sumber: DATAAKSI — dataset_fiskal_97_hipoFull.csv. "
    "Interpretasi indikator mengikuti definisi dan satuan "
    "pada data sumber."
)

# ============================================================
# 7. KPI CARDS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Observasi",
        f"{len(filtered):,}",
    )

with c2:
    st.metric(
        "Wilayah",
        f"{filtered['Kota'].nunique():,}",
    )

with c3:
    st.metric(
        "Rentang Tahun",
        f"{int(filtered['Tahun'].min())}–"
        f"{int(filtered['Tahun'].max())}",
    )

with c4:
    st.metric(
        "Indikator Numerik",
        f"{len(numeric_cols):,}",
    )

st.markdown("---")


# ============================================================
# 8. MAIN TABS
# ============================================================

tabs = st.tabs([
    "OVERVIEW",
    "FISCAL",
    "EXPENDITURE",
    "DEVELOPMENT",
    "VISUALIZATION",
    "CORRELATION",
    "OLS REGRESSION",
    "DIAGNOSTICS",
    "DATA & EXPORT",
])


# ============================================================
# TAB 1: OVERVIEW
# ============================================================

with tabs[0]:
    st.subheader("Dataset Overview")

    left, right = st.columns(2)

    with left:
        st.markdown("**Jumlah observasi per tahun**")

        year_counts = (
            filtered.groupby("Tahun")
            .size()
            .reset_index(name="Jumlah Observasi")
        )

        fig = px.bar(
            year_counts,
            x="Tahun",
            y="Jumlah Observasi",
            text="Jumlah Observasi",
            title="Distribusi Observasi Tahunan",
        )
        fig.update_layout(
            template="plotly_white",
            xaxis_title="Tahun",
            yaxis_title="Jumlah",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("**Statistik deskriptif**")
        st.dataframe(
            filtered[numeric_cols].describe().T,
            use_container_width=True,
        )

    st.subheader("Kelengkapan Data")

    missing = (
        filtered.isna()
        .sum()
        .rename("Missing")
        .reset_index()
        .rename(columns={"index": "Variabel"})
    )

    missing["Persentase Missing (%)"] = (
        missing["Missing"] / len(filtered) * 100
    ).round(2)

    st.dataframe(missing, use_container_width=True)

    st.subheader("Rata-rata Indikator per Tahun")

    chosen_overview = st.selectbox(
        "Indikator",
        numeric_cols,
        key="overview_indicator",
    )

    annual = (
        filtered.groupby("Tahun")[chosen_overview]
        .mean()
        .reset_index()
    )

    fig = px.line(
        annual,
        x="Tahun",
        y=chosen_overview,
        markers=True,
        title=f"Tren Tahunan: {chosen_overview}",
    )
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 2: FISCAL ANALYSIS
# ============================================================

with tabs[1]:
    st.subheader("Fiscal Capacity & Revenue")

    available_fiscal = fiscal_cols or numeric_cols

    fiscal_indicator = st.selectbox(
        "Pilih indikator fiskal",
        available_fiscal,
        key="fiscal_indicator",
    )

    city_annual = (
        filtered.groupby(["Tahun", "Kota"], as_index=False)
        [fiscal_indicator].mean()
    )

    fig = px.line(
        city_annual,
        x="Tahun",
        y=fiscal_indicator,
        color="Kota",
        markers=True,
        title=f"Tren {fiscal_indicator} antarwilayah",
    )
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Perbandingan Fiskal Antarwilayah")

    ranking_year = st.selectbox(
        "Tahun untuk ranking",
        sorted(
            filtered["Tahun"].dropna().astype(int).unique()
        ),
        key="fiscal_rank_year",
    )

    rank_df = filtered[
        filtered["Tahun"] == ranking_year
    ].groupby("Kota", as_index=False)[fiscal_indicator].mean()

    rank_df = rank_df.sort_values(
        fiscal_indicator, ascending=False
    )

    fig = px.bar(
        rank_df,
        x=fiscal_indicator,
        y="Kota",
        orientation="h",
        title=f"Ranking {fiscal_indicator} ({ranking_year})",
    )
    fig.update_layout(
        template="plotly_white",
        yaxis={"categoryorder": "total ascending"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Ranking berdasarkan nilai indikator, bukan otomatis "
        "berdasarkan kualitas fiskal. Nilai yang lebih tinggi "
        "belum tentu lebih baik."
    )


# ============================================================
# TAB 3: EXPENDITURE ANALYSIS
# ============================================================

with tabs[2]:
    st.subheader("Belanja Modal Sektoral")

    available_spending = spending_cols or numeric_cols

    expenditure_vars = st.multiselect(
        "Pilih indikator belanja",
        available_spending,
        default=available_spending[:min(5, len(available_spending))],
        key="expenditure_vars",
    )

    if expenditure_vars:
        expenditure_annual = (
            filtered.groupby("Tahun")[expenditure_vars]
            .mean()
            .reset_index()
        )

        expenditure_long = expenditure_annual.melt(
            id_vars="Tahun",
            var_name="Indikator",
            value_name="Nilai",
        )

        fig = px.line(
            expenditure_long,
            x="Tahun",
            y="Nilai",
            color="Indikator",
            markers=True,
            title="Tren Belanja Modal Sektoral",
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Komposisi Belanja Berdasarkan Rata-rata")

        composition = (
            filtered[expenditure_vars]
            .mean()
            .rename("Rata-rata")
            .reset_index()
            .rename(columns={"index": "Sektor"})
        )

        fig = px.bar(
            composition,
            x="Sektor",
            y="Rata-rata",
            title="Rata-rata Nilai per Indikator Belanja",
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "Perbandingan komposisi ini menggunakan nilai rata-rata "
            "kolom. Pastikan satuan dan definisi tiap variabel "
            "sebanding sebelum menafsirkan proporsinya."
        )
    else:
        st.warning("Pilih setidaknya satu indikator.")


# ============================================================
# TAB 4: DEVELOPMENT INDICATORS
# ============================================================

with tabs[3]:
    st.subheader("Regional Development Outcomes")

    available_social = social_cols or numeric_cols

    development_vars = st.multiselect(
        "Pilih indikator pembangunan",
        available_social,
        default=available_social[:min(4, len(available_social))],
        key="development_vars",
    )

    if development_vars:
        development_long = filtered.melt(
            id_vars=["Kota", "Tahun"],
            value_vars=development_vars,
            var_name="Indikator",
            value_name="Nilai",
        )

        fig = px.line(
            development_long,
            x="Tahun",
            y="Nilai",
            color="Kota",
            facet_col="Indikator",
            facet_col_wrap=2,
            markers=True,
            title="Tren Indikator Pembangunan",
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Perbandingan Antarwilayah")

        dev_rank_indicator = st.selectbox(
            "Indikator ranking",
            development_vars,
            key="dev_rank_indicator",
        )

        dev_rank_year = st.selectbox(
            "Tahun ranking",
            sorted(
                filtered["Tahun"].dropna().astype(int).unique()
            ),
            key="dev_rank_year",
        )

        dev_rank = filtered[
            filtered["Tahun"] == dev_rank_year
        ].groupby("Kota", as_index=False)[
            dev_rank_indicator
        ].mean()

        dev_rank = dev_rank.sort_values(
            dev_rank_indicator,
            ascending=False,
        )

        fig = px.bar(
            dev_rank,
            x=dev_rank_indicator,
            y="Kota",
            orientation="h",
            title=f"{dev_rank_indicator} — {dev_rank_year}",
        )
        fig.update_layout(
            template="plotly_white",
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Untuk kemiskinan dan pengangguran, nilai lebih rendah "
            "biasanya diinginkan; untuk IPM, nilai lebih tinggi "
            "biasanya diinginkan. Perhatikan definisi indikator."
        )


# ============================================================
# TAB 5: GENERAL VISUALIZATION
# ============================================================

with tabs[4]:
    st.subheader("Interactive Visualization Lab")

    chart_type = st.selectbox(
        "Jenis grafik",
        [
            "Line Chart",
            "Bar Chart",
            "Scatter Plot",
            "Histogram",
            "Box Plot",
        ],
    )

    x_var = st.selectbox(
        "Variabel X",
        numeric_cols,
        key="general_x",
    )

    y_var = st.selectbox(
        "Variabel Y",
        numeric_cols,
        index=min(1, len(numeric_cols) - 1),
        key="general_y",
    )

    color_options = ["Tidak ada"] + [
        c for c in ["Kota", "Tahun"] if c in filtered.columns
    ]

    color_var = st.selectbox(
        "Warna / kelompok",
        color_options,
    )

    color_arg = None if color_var == "Tidak ada" else color_var

    if chart_type == "Line Chart":
        fig = px.line(
            filtered.sort_values("Tahun"),
            x="Tahun",
            y=y_var,
            color=color_arg,
            markers=True,
        )

    elif chart_type == "Bar Chart":
        bar_data = filtered.groupby(
            color_arg if color_arg else "Tahun",
            as_index=False,
        )[y_var].mean()

        fig = px.bar(
            bar_data,
            x=color_arg if color_arg else "Tahun",
            y=y_var,
        )

    elif chart_type == "Scatter Plot":
        fig = px.scatter(
            filtered,
            x=x_var,
            y=y_var,
            color=color_arg,
            hover_data=["Kota", "Tahun"],
            trendline="ols",
        )

    elif chart_type == "Histogram":
        fig = px.histogram(
            filtered,
            x=x_var,
            color=color_arg,
            marginal="box",
        )

    else:
        fig = px.box(
            filtered,
            x=color_arg,
            y=y_var,
            points="all",
        )

    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 6: CORRELATION ANALYSIS
# ============================================================

with tabs[5]:
    st.subheader("Correlation Matrix")

    corr_vars = st.multiselect(
        "Pilih variabel untuk korelasi",
        numeric_cols,
        default=numeric_cols[:min(10, len(numeric_cols))],
        key="corr_vars",
    )

    corr_method = st.radio(
        "Metode korelasi",
        ["pearson", "spearman", "kendall"],
        horizontal=True,
    )

    if len(corr_vars) >= 2:
        corr = filtered[corr_vars].corr(
            method=corr_method
        )

        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
            title=f"Correlation Matrix — {corr_method.title()}",
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Korelasi terhadap Target")

        target_corr = st.selectbox(
            "Target korelasi",
            corr_vars,
            key="target_corr",
        )

        corr_target = (
            corr[target_corr]
            .drop(target_corr)
            .sort_values()
            .reset_index()
        )
        corr_target.columns = ["Variabel", "Korelasi"]

        fig = px.bar(
            corr_target,
            x="Korelasi",
            y="Variabel",
            orientation="h",
            color="Korelasi",
            color_continuous_scale="RdBu_r",
            range_color=[-1, 1],
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Korelasi menunjukkan asosiasi, bukan bukti hubungan "
            "kausal. Periksa struktur panel, tren, dan faktor "
            "perancu sebelum membuat kesimpulan."
        )
    else:
        st.info("Pilih minimal dua variabel numerik.")


# ============================================================
# TAB 7: OLS REGRESSION
# ============================================================

with tabs[6]:
    st.subheader("Exploratory OLS Regression")

    st.warning(
        "Model ini merupakan regresi eksploratif. Data antarkota "
        "dan antarwaktu dapat memiliki struktur panel; OLS biasa "
        "tidak otomatis mengatasi heterogenitas wilayah, "
        "autokorelasi, atau endogenitas."
    )

    default_target = (
        "IPM" if "IPM" in numeric_cols else numeric_cols[0]
    )

    target = st.selectbox(
        "Dependent variable (Y)",
        numeric_cols,
        index=numeric_cols.index(default_target),
        key="ols_target",
    )

    predictor_options = [
        c for c in numeric_cols if c != target
    ]

    default_predictors = [
        c for c in [
            "PAD", "ICT", "bm_pendidikan",
            "bm_kesehatan", "DAU_Grant"
        ]
        if c in predictor_options
    ]

    predictors = st.multiselect(
        "Independent variables (X)",
        predictor_options,
        default=default_predictors,
        key="ols_predictors",
    )

    add_city_fe = st.checkbox(
        "Tambahkan dummy fixed effect Kota",
        value=False,
    )

    add_year_fe = st.checkbox(
        "Tambahkan dummy fixed effect Tahun",
        value=False,
    )

    use_hc3 = st.checkbox(
        "Gunakan standard errors HC3",
        value=True,
    )

    if st.button("🚀 Estimate OLS Model", type="primary"):
        if not predictors:
            st.error("Pilih minimal satu variabel independen.")
        else:
            model_cols = [target] + predictors + ["Kota", "Tahun"]
            model_df = filtered[model_cols].dropna().copy()

            X = model_df[predictors].astype(float)
            y = model_df[target].astype(float)

            if add_city_fe:
                city_dummies = pd.get_dummies(
                    model_df["Kota"],
                    prefix="Kota",
                    drop_first=True,
                    dtype=float,
                )
                X = pd.concat(
                    [X.reset_index(drop=True),
                     city_dummies.reset_index(drop=True)],
                    axis=1,
                )

            if add_year_fe:
                year_dummies = pd.get_dummies(
                    model_df["Tahun"].astype(str),
                    prefix="Tahun",
                    drop_first=True,
                    dtype=float,
                )
                X = pd.concat(
                    [X.reset_index(drop=True),
                     year_dummies.reset_index(drop=True)],
                    axis=1,
                )

            X = sm.add_constant(X, has_constant="add")

            if len(model_df) <= X.shape[1]:
                st.error(
                    "Observasi tidak cukup untuk jumlah parameter. "
                    "Kurangi prediktor atau dummy fixed effect."
                )
            elif X.nunique().le(1).any():
                st.error(
                    "Terdapat variabel konstan. Ubah filter atau "
                    "hapus variabel yang tidak bervariasi."
                )
            else:
                try:
                    ols_model = sm.OLS(y, X).fit(
                        cov_type="HC3" if use_hc3 else "nonrobust"
                    )

                    st.session_state["ols_model"] = ols_model
                    st.session_state["ols_X"] = X
                    st.session_state["ols_y"] = y
                    st.session_state["ols_df"] = model_df
                    st.session_state["ols_target_name"] = target
                    st.session_state["ols_predictors"] = predictors

                except Exception as e:
                    st.error(f"Estimasi gagal: {e}")

    if "ols_model" in st.session_state:
        model = st.session_state["ols_model"]

        if model.nobs == 0:
            st.warning("Tidak ada observasi model.")
        else:
            m1, m2, m3, m4 = st.columns(4)

            m1.metric("Observations", f"{int(model.nobs):,}")
            m2.metric("R-squared", f"{model.rsquared:.4f}")
            m3.metric("Adjusted R²", f"{model.rsquared_adj:.4f}")
            m4.metric("Prob (F-statistic)", f"{model.f_pvalue:.4g}")

            st.subheader("Regression Coefficients")

            coef_df = pd.DataFrame({
                "Variable": model.params.index,
                "Coefficient": model.params.values,
                "Std. Error": model.bse.values,
                "t-statistic": model.tvalues.values,
                "p-value": model.pvalues.values,
                "CI Lower": model.conf_int()[0].values,
                "CI Upper": model.conf_int()[1].values,
            })

            st.dataframe(
                coef_df.round(5),
                use_container_width=True,
            )

            coef_plot = coef_df[
                coef_df["Variable"] != "const"
            ].copy()

            if not coef_plot.empty:
                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=coef_plot["Coefficient"],
                        y=coef_plot["Variable"],
                        mode="markers",
                        error_x=dict(
                            type="data",
                            symmetric=False,
                            array=(
                                coef_plot["CI Upper"]
                                - coef_plot["Coefficient"]
                            ),
                            arrayminus=(
                                coef_plot["Coefficient"]
                                - coef_plot["CI Lower"]
                            ),
                        ),
                    )
                )

                fig.add_vline(x=0, line_dash="dash")
                fig.update_layout(
                    title="Coefficient Plot with 95% Confidence Intervals",
                    xaxis_title="Coefficient",
                    yaxis_title="Variable",
                    template="plotly_white",
                )
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Full OLS Summary")
            st.code(model.summary().as_text())

            st.download_button(
                "⬇️ Download Coefficients CSV",
                coef_df.to_csv(index=False).encode("utf-8"),
                file_name="ols_coefficients.csv",
                mime="text/csv",
            )


# ============================================================
# TAB 8: REGRESSION DIAGNOSTICS
# ============================================================

with tabs[7]:
    st.subheader("OLS Diagnostics")

    if "ols_model" not in st.session_state:
        st.info(
            "Estimasi model terlebih dahulu pada tab "
            "OLS REGRESSION."
        )
    else:
        model = st.session_state["ols_model"]

        residuals = np.asarray(model.resid)
        fitted = np.asarray(model.fittedvalues)

        left, right = st.columns(2)

        with left:
            fig = px.scatter(
                x=fitted,
                y=residuals,
                labels={
                    "x": "Fitted Values",
                    "y": "Residuals",
                },
                title="Residuals vs Fitted",
            )
            fig.add_hline(y=0, line_dash="dash")
            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with right:
            qq = sm.ProbPlot(residuals)
            theoretical = qq.theoretical_quantiles
            ordered_residuals = np.sort(residuals)

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=theoretical,
                    y=ordered_residuals,
                    mode="markers",
                    name="Residuals",
                )
            )

            # Garis referensi QQ plot
            q1, q3 = np.percentile(ordered_residuals, [25, 75])
            t1, t3 = np.percentile(theoretical, [25, 75])

            if t3 != t1:
                slope = (q3 - q1) / (t3 - t1)
                intercept = q1 - slope * t1

                fig.add_trace(
                    go.Scatter(
                        x=theoretical,
                        y=intercept + slope * theoretical,
                        mode="lines",
                        name="Reference line",
                    )
                )

            fig.update_layout(
                title="Q-Q Plot",
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Ordered Residuals",
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Diagnostic Test Results")

        residual_series = pd.Series(residuals)
        exog = np.asarray(model.model.exog)

        diag_results = []

        # Breusch-Pagan
        try:
            bp = het_breuschpagan(
                residual_series, exog
            )
            diag_results.append({
                "Test": "Breusch-Pagan",
                "Statistic": bp[0],
                "p-value": bp[1],
                "Interpretation": (
                    "Indikasi heteroskedastisitas"
                    if bp[1] < 0.05
                    else "Tidak menolak homoskedastisitas"
                ),
            })
        except Exception:
            pass

        # White test requires enough variation and observations
        try:
            white = het_white(
                residual_series, exog
            )
            diag_results.append({
                "Test": "White",
                "Statistic": white[0],
                "p-value": white[1],
                "Interpretation": (
                    "Indikasi heteroskedastisitas"
                    if white[1] < 0.05
                    else "Tidak menolak homoskedastisitas"
                ),
            })
        except Exception:
            pass

        # Jarque-Bera
        try:
            jb_stat, jb_p, skew, kurtosis = jarque_bera(
                residual_series
            )
            diag_results.append({
                "Test": "Jarque-Bera",
                "Statistic": jb_stat,
                "p-value": jb_p,
                "Interpretation": (
                    "Residual tidak normal menurut uji"
                    if jb_p < 0.05
                    else "Tidak menolak normalitas residual"
                ),
            })
        except Exception:
            pass

        # Durbin-Watson
        try:
            dw = durbin_watson(residual_series)
            diag_results.append({
                "Test": "Durbin-Watson",
                "Statistic": dw,
                "p-value": np.nan,
                "Interpretation": (
                    "Periksa autokorelasi; nilai sekitar 2 "
                    "sering menjadi indikasi awal tanpa "
                    "autokorelasi orde pertama"
                ),
            })
        except Exception:
            pass

        if diag_results:
            st.dataframe(
                pd.DataFrame(diag_results).round(5),
                use_container_width=True,
            )

        st.caption(
            "Catatan: Durbin-Watson bukan uji lengkap untuk "
            "autokorelasi panel. Dummy wilayah/tahun juga tidak "
            "menggantikan standard errors yang robust terhadap "
            "pengelompokan wilayah."
        )


# ============================================================
# TAB 9: DATA TABLE & EXPORT
# ============================================================

with tabs[8]:
    st.subheader("Filtered Dataset")

    search_city = st.text_input(
        "Cari nama wilayah",
        placeholder="Contoh: Kota Makassar",
    )

    export_df = filtered.copy()

    if search_city:
        export_df = export_df[
            export_df["Kota"].str.contains(
                search_city,
                case=False,
                na=False,
            )
        ]

    st.write(
        f"Menampilkan {len(export_df):,} baris "
        f"dan {len(export_df.columns):,} kolom."
    )

    st.dataframe(
        export_df,
        use_container_width=True,
        height=500,
    )

    csv_data = export_df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "⬇️ Download Filtered Data (CSV)",
        data=csv_data,
        file_name="dataset_fiskal_filtered.csv",
        mime="text/csv",
    )

    # Excel export
    try:
        excel_buffer = io.BytesIO()

        with pd.ExcelWriter(
            excel_buffer,
            engine="openpyxl",
        ) as writer:
            export_df.to_excel(
                writer,
                sheet_name="Filtered Data",
                index=False,
            )

            export_df[numeric_cols].describe().T.to_excel(
                writer,
                sheet_name="Descriptive Stats",
            )

        st.download_button(
            "⬇️ Download Excel Report",
            data=excel_buffer.getvalue(),
            file_name="fiscal_analytics_report.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
        )

    except Exception as e:
        st.warning(
            "Ekspor Excel memerlukan openpyxl. "
            f"Detail: {e}"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption(
    "Fiscal & Regional Development Analytics | DATAAKSI | "
    "Exploratory analysis — interpret results with care."
)
