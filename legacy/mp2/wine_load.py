import pandas as pd  
import plotly.express as px
import numpy as np


# Load the two CSVs into DataFrames
red_wine = pd.read_excel("data/winequality-red.xlsx", skiprows=1)
white_wine = pd.read_excel("data/winequality-white.xlsx", skiprows=1)

"""
# Print some info about each dataset
print("Red wine dataset:")
print(red_wine.head())   # show first 5 rows
print()
print("White wine dataset:")
print(white_wine.head())
"""

red_wine["type"] = "Red wine"
white_wine["type"] = "White wine"

all_wine = pd.concat([red_wine, white_wine], ignore_index=True)

"""
print("Merged dataset:")
print(all_wine.head())
print()
print("\nshape of merged dataset:",all_wine.shape)
"""

"""
# columns should match and be in the expected order
print(all_wine.columns.tolist())

# spot-check a couple rows; pH should be ~2.7–4.0, sulphates ~0.3–0.7, quality int 3–9
print(all_wine.head(2))
print(all_wine.tail(2))

# confirm the type labels
print(all_wine["type"].unique())

"""
"""
# --- Basic exploration ---

# How many rows/columns?
print("Shape:", all_wine.shape)

# Summary statistics
print("\nSummary statistics:")
print(all_wine.describe())

# Average quality by wine type
print("\nAverage quality by type:")
print(all_wine.groupby("type")["quality"].mean())

# Average alcohol by wine type
print("\nAverage alcohol by type:")
print(all_wine.groupby("type")["alcohol"].mean())

# Average residual sugar by wine type
print("\nAverage residual sugar by type:")
print(all_wine.groupby("type")["residual sugar"].mean())

"""
"""

# Histogram of alcohol by type
fig = px.histogram(
    all_wine, 
    x="alcohol", 
    color="type",
    nbins=30, 
    barmode="overlay",
    title="Alcohol Distribution by Wine Type"
)
fig.show()

# Boxplot of residual sugar by type
fig = px.box(
    all_wine, 
    x="type", 
    y="residual sugar", 
    title="Residual Sugar by Wine Type"
)
fig.show()

# Boxplot of quality by type
fig = px.box(
    all_wine, 
    x="type", 
    y="quality", 
    title="Wine Quality by Type"
)
fig.show()

"""

# 1. Does higher alcohol always mean higher quality?
print("\nAverage quality by alcohol quartile:")
print(all_wine.groupby(pd.qcut(all_wine["alcohol"], 4))["quality"].mean())

# 2. Are sweeter wines (more residual sugar) rated higher or lower?
print("\nAverage quality by residual sugar quartile:")
print(all_wine.groupby(pd.qcut(all_wine["residual sugar"], 4))["quality"].mean())

# 3. Is there a 'sweet spot' of acidity/pH for higher ratings?
print("\nAverage quality by pH quartile:")
print(all_wine.groupby(pd.qcut(all_wine["pH"], 4))["quality"].mean())

# 4. Which wine type is more consistent (less variable) in quality?
print("\nQuality variability (std dev) by type:")
print(all_wine.groupby("type")["quality"].std())

# 5. What ranges of sulfur dioxide are typical?
print("\nSulfur dioxide summary (free & total):")
print(all_wine[["free sulfur dioxide","total sulfur dioxide"]].describe())

# Step 10 – Binning pH into groups

# Split pH into 5 groups
bins_5 = pd.cut(all_wine["pH"], bins=5)
print("\nCounts of wines in 5 pH groups:")
print(all_wine.groupby(bins_5).size())

# Split pH into 10 groups
bins_10 = pd.cut(all_wine["pH"], bins=10)
print("\nCounts of wines in 10 pH groups:")
print(all_wine.groupby(bins_10).size())

# --- Step 11: Correlations & Heatmap ---



# Use only numeric columns
num = all_wine.select_dtypes(include=[np.number]).copy()

# Correlation matrix (Pearson)
corr = num.corr(method="pearson")

print("\nCorrelation with quality (sorted):")
q_corr = corr["quality"].sort_values(ascending=False)
print(q_corr)

# Which has the strongest/weakest relationship with quality?
strongest_feat = q_corr.drop(labels=["quality"]).abs().idxmax()
strongest_val  = q_corr[strongest_feat]
weakest_feat   = q_corr.drop(labels=["quality"]).abs().idxmin()
weakest_val    = q_corr[weakest_feat]

print(f"\nStrongest with quality: {strongest_feat} ({strongest_val:+.3f})")
print(f"Weakest with quality:  {weakest_feat} ({weakest_val:+.3f})")

# Find other strongly correlated pairs (exclude 'quality', avoid duplicates, threshold = 0.6)
pairs = []
cols = [c for c in corr.columns if c != "quality"]
thr = 0.60
for i, a in enumerate(cols):
    for b in cols[i+1:]:
        val = corr.loc[a, b]
        if abs(val) >= thr:
            pairs.append((a, b, float(val)))

pairs_sorted = sorted(pairs, key=lambda t: -abs(t[2]))
print("\nHighly correlated non-quality pairs (|r| ≥ 0.60):")
for a, b, v in pairs_sorted:
    print(f"{a} ↔ {b}: r = {v:+.3f}")

# Plotly heatmap of the full correlation matrix
fig = px.imshow(
    corr,
    text_auto=True,
    aspect="auto",
    title="Correlation Matrix (Pearson)"
)
fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
fig.show()

# --- Step 12: Outlier Detection and Removal ---

def find_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    mask = (df[column] < lower) | (df[column] > upper)
    return df[mask]

# Check each numeric column for outliers
numeric_cols = all_wine.select_dtypes(include="number").columns
print("\nOutlier counts by feature:")
for col in numeric_cols:
    outliers = find_outliers_iqr(all_wine, col)
    if len(outliers) > 0:
        print(f"{col}: {len(outliers)} outliers")

# Example: look at outliers in 'residual sugar'
print("\nResidual sugar outliers (first 5):")
print(find_outliers_iqr(all_wine, "residual sugar").head())

# Remove outliers (here we apply it to ALL numeric columns)
cleaned = all_wine.copy()
for col in numeric_cols:
    outliers = find_outliers_iqr(cleaned, col)
    cleaned = cleaned.drop(outliers.index)

print(f"\nShape before: {all_wine.shape}, after removing outliers: {cleaned.shape}")
