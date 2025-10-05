import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ---------- 1) Load data again ----------
df = pd.read_csv("data/hr.csv", encoding="utf-8-sig")
df = df.drop(columns=["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"], errors="ignore")

# Targets
y = df["MonthlyIncome"]
X = df.drop(columns=["MonthlyIncome", "Attrition"])

# Split numeric vs categorical
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

print("Numeric features:", num_cols[:5], "...")
print("Categorical features:", cat_cols[:5], "...")

# ---------- 2) Preprocessing ----------
# - Numeric: scale to similar range
# - Categorical: one-hot encode
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

# ---------- 3) Train/test split ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- 4) Build pipeline ----------
model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("regressor", LinearRegression())
])

# ---------- 5) Train ----------
model.fit(X_train, y_train)

# ---------- 6) Evaluate ----------
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)   # mean squared error
rmse = np.sqrt(mse)                        # root mean squared error
r2 = r2_score(y_test, y_pred)

print(f"\nRegression Results:")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.3f}")
