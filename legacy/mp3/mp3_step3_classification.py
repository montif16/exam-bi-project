import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# ---------- 1) Load and clean ----------
df = pd.read_csv("data/hr.csv", encoding="utf-8-sig")
df = df.drop(columns=["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"], errors="ignore")

# Target (Attrition → 0/1)
y = df["Attrition"].map({"Yes": 1, "No": 0})
X = df.drop(columns=["Attrition", "MonthlyIncome"])  # drop targets, use rest as features

# Split numeric vs categorical
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

print("Numeric features:", num_cols[:5], "...")
print("Categorical features:", cat_cols[:5], "...")

# ---------- 2) Preprocessing ----------
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

# ---------- 3) Train/test split ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---------- 4) Build pipeline ----------
clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("classifier", LogisticRegression(max_iter=1000))
])

# ---------- 5) Train ----------
clf.fit(X_train, y_train)

# ---------- 6) Evaluate ----------
y_pred = clf.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
