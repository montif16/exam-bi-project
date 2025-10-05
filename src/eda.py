import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def describe(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe(include="all").T

def plot_hist(df: pd.DataFrame, column: str):
    ax = df[column].dropna().plot(kind="hist", bins=30)
    ax.set_title(f"Histogram: {column}")
    ax.set_xlabel(column)
    return ax

def plot_scatter(df: pd.DataFrame, x: str, y: str):
    ax = df.plot(kind="scatter", x=x, y=y)
    ax.set_title(f"Scatter: {x} vs {y}")
    return ax

# ---- New: correlation heatmap (numeric-only)
def corr_heatmap(df: pd.DataFrame, top_n: int = 15):
    num = df.select_dtypes(include=[np.number, "bool"]).copy()
    # keep only columns with some variance
    num = num.loc[:, num.nunique(dropna=True) > 1]
    if "G3" in num.columns:
        # show strongest correlations with G3
        corrs = num.corr(numeric_only=True)["G3"].abs().sort_values(ascending=False).head(top_n).index
        view = num[corrs]
    else:
        # fallback to top_n most varying features
        var = num.var(numeric_only=True).sort_values(ascending=False).head(top_n).index
        view = num[var]
    corr = view.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.index)
    ax.set_title("Correlation heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig

# ---- New: bar chart of mean G3 by a discrete column
def bar_mean_g3_by(df: pd.DataFrame, by_col: str = "studytime"):
    if "G3" not in df.columns or by_col not in df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "G3 or selected column not found", ha="center")
        return fig
    # treat by_col as categorical (bin numeric with few bins if needed)
    series = df[by_col]
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > 12:
        # bin into quartiles
        series = pd.qcut(series, 4, duplicates="drop")
        grouped = df.groupby(series)["G3"].mean()
        xlabel = f"{by_col} (quartiles)"
    else:
        grouped = df.groupby(series)["G3"].mean()
        xlabel = by_col

    fig, ax = plt.subplots()
    grouped.plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean G3")
    ax.set_xlabel(xlabel)
    ax.set_title(f"Average G3 by {by_col}")
    return fig
