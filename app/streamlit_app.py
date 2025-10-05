from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]  # project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
from src.data_ingestion import build_dataset, PROCESSED
from src.eda import describe, plot_hist, plot_scatter
from src.models import train_classifier, train_regressor, train_cluster

st.set_page_config(page_title="BI Exam Prototype", layout="wide")
st.title("BI/AI Exam – Student Performance")

@st.cache_data
def get_df():
    try:
        df = pd.read_csv(PROCESSED / "dataset_clean.csv")
    except FileNotFoundError:
        df = build_dataset()
    return df

df = get_df()

# Heuristics for default targets & features (Student Performance)
all_cols = df.columns.tolist()
default_class_target = "Pass" if "Pass" in df.columns else None
default_reg_target = "G3" if "G3" in df.columns else None

# Try a small sensible numeric feature set commonly present in student datasets
candidate_features = [c for c in ["age", "studytime", "failures", "absences", "Medu", "Fedu"] if c in df.columns]
# Add one-hot columns that are likely present
candidate_features += [c for c in all_cols if c.startswith("sex_") or c.startswith("address_") or c.startswith("famsize_")]

# Fallback to any numeric columns if the above are missing
if len(candidate_features) < 4:
    candidate_features = df.select_dtypes(include=["number", "bool", "int", "float"]).columns.tolist()
    # Don't include targets as features
    candidate_features = [c for c in candidate_features if c not in {"Pass", "G3"}]

st.subheader("Data Preview")
st.dataframe(df.head(20))

with st.expander("Descriptive Statistics"):
    st.dataframe(describe(df))

st.subheader("Quick Plots")
cols = list(df.columns)
if cols:
    c1, c2 = st.columns(2)
    with c1:
        # Prefer numeric column for histogram defaults
        num_cols = df.select_dtypes(include=["number", "bool", "int", "float"]).columns.tolist() or cols
        col = st.selectbox("Histogram column", num_cols, index=0)
        fig1 = plot_hist(df, col).get_figure(); st.pyplot(fig1); fig1.clf()
    with c2:
        x = st.selectbox("Scatter X", num_cols, index=0)
        y = st.selectbox("Scatter Y", num_cols, index=1 if len(num_cols) > 1 else 0)
        fig2 = plot_scatter(df, x, y).get_figure(); st.pyplot(fig2); fig2.clf()
        
# ---- New: stronger EDA visuals
st.subheader("Exploratory Analysis")

c_heat, c_bar = st.columns([1,1])

with c_heat:
    st.caption("Correlation heatmap (top features)")
    try:
        from src.eda import corr_heatmap
        fig_hm = corr_heatmap(df)
        st.pyplot(fig_hm)
    except Exception as e:
        st.warning(f"Heatmap error: {e}")

with c_bar:
    st.caption("Average G3 by factor")
    from src.eda import bar_mean_g3_by
    # sensible defaults if present
    default_factor = "studytime" if "studytime" in df.columns else ("failures" if "failures" in df.columns else None)
    factors = [c for c in df.columns if c in {"studytime","failures","absences","Medu","Fedu"}] or df.columns.tolist()
    by = st.selectbox("Group by", factors, index=(factors.index(default_factor) if default_factor in factors else 0))
    fig_bar = bar_mean_g3_by(df, by_col=by)
    st.pyplot(fig_bar)

st.subheader("Models")
t1, t2, t3 = st.tabs(["Classification", "Regression", "Clustering"])

with t1:
    st.write("Binary classification (default: Pass)")
    features = st.multiselect("Features", df.columns.tolist(), default=candidate_features)
    target = st.selectbox("Target (binary/class)", df.columns.tolist(), index=(df.columns.get_loc(default_class_target) if default_class_target in df.columns else 0))
    if st.button("Train Classifier"):
        use = df.dropna(subset=list(set(features + [target])))
        st.success(train_classifier(use, features, target))

with t2:
    st.write("Regression (default: G3 final grade)")
    features_r = st.multiselect("Features", df.columns.tolist(), default=candidate_features, key="regf")
    # default to G3 where available
    idx = (df.columns.get_loc(default_reg_target) if default_reg_target in df.columns else 0)
    target_r = st.selectbox("Target (numeric)", df.columns.tolist(), index=idx, key="regt")
    if st.button("Train Regressor"):
        use = df.dropna(subset=list(set(features_r + [target_r])))
        st.success(train_regressor(use, features_r, target_r))

with t3:
    st.write("KMeans clustering")
    features_c = st.multiselect("Features", df.columns.tolist(), default=candidate_features, key="cluf")
    k = st.slider("k", 2, 10, 3)
    if st.button("Train Cluster"):
        use = df.dropna(subset=features_c)
        st.success(train_cluster(use, features_c, k))

st.caption("Dataset: Student Performance (Maths). Pass = (G3 ≥ 10). Categorical features are one-hot encoded.")
