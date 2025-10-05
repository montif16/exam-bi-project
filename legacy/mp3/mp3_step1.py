import os
import pandas as pd
import numpy as np

# ---------- 1) Load ----------
path = "data/hr.csv"
if not os.path.exists(path):
    raise FileNotFoundError("Put the IBM HR dataset as data/hr.csv")

# Read CSV (handle UTF-8 BOM if present)
df = pd.read_csv(path, encoding="utf-8-sig")
df.columns = df.columns.str.strip()  # tidy headers

print("Loaded:", path)
print("Shape:", df.shape)
print("\nColumns:", list(df.columns))

# ---------- 2) Quick checks ----------
print("\nDtypes:")
print(df.dtypes)

print("\nMissing values (top 20):")
missing = df.isna().sum().sort_values(ascending=False)
print(missing.head(20))

print("\nFirst 5 rows:")
print(df.head())

# ---------- 3) Identify junk/ID/constant columns ----------
const_cols = [c for c in df.columns if df[c].nunique(dropna=False) == 1]
id_like = [c for c in df.columns if "id" in c.lower() or "employee" in c.lower()]

print("\nConstant columns (nunique==1):", const_cols)
print("ID-like columns:", id_like)

# Known constants/IDs in this dataset
DROP_CANDIDATES = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]
to_drop = [c for c in DROP_CANDIDATES if c in df.columns]
print("\nPlanned drops (if present):", to_drop)

df_clean = df.drop(columns=to_drop, errors="ignore")

# ---------- 4) Prepare targets for later ----------
# Regression target (Step 2)
if "MonthlyIncome" not in df_clean.columns:
    raise KeyError("MonthlyIncome column not found. Check your CSV columns.")
y_income = df_clean["MonthlyIncome"]

# Classification target (Step 3): Attrition -> 1/0
if "Attrition" not in df_clean.columns:
    raise KeyError("Attrition column not found. Check your CSV columns.")
y_attrition = df_clean["Attrition"].map({"Yes": 1, "No": 0})
if y_attrition.isna().any():
    raise ValueError("Attrition column has unexpected values; expected 'Yes'/'No'.")

print("\nAttrition distribution (0=No, 1=Yes):")
print(y_attrition.value_counts())

# Features (X_base) = all except targets
X_base = df_clean.drop(columns=["MonthlyIncome", "Attrition"])

# Split numeric vs categorical (for later preprocessing)
num_cols = X_base.select_dtypes(include=np.number).columns.tolist()
cat_cols = X_base.select_dtypes(exclude=np.number).columns.tolist()

print("\nNumeric feature columns (sample):", num_cols[:10])
print("Categorical feature columns (sample):", cat_cols[:10])

print("\nBasic numeric summary:")
print(X_base.describe().T.head(12))
