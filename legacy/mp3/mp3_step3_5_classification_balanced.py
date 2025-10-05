import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score

# ---------- 1) Load and basic clean ----------
df = pd.read_csv("data/hr.csv", encoding="utf-8-sig")
df = df.drop(columns=["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"], errors="ignore")

# Target (Attrition -> 1 if quits, 0 if stays)
y = df["Attrition"].map({"Yes": 1, "No": 0})
X = df.drop(columns=["Attrition", "MonthlyIncome"])  # keep it same as Step 3

# Split numeric vs categorical
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

print("Numeric features:", num_cols[:5], "...")
print("Categorical features:", cat_cols[:5], "...")

# ---------- 2) Preprocessing ----------
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

# ---------- 3) Train/test split (stratified keeps class ratio) ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- 4) Model with class weighting ----------
clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
])

# ---------- 5) Train ----------
clf.fit(X_train, y_train)

# ---------- 6) Evaluate ----------
y_pred = clf.predict(X_test)
# Probabilities for extra metrics (ROC-AUC / PR-AUC)
if hasattr(clf.named_steps["classifier"], "predict_proba"):
    y_proba = clf.predict_proba(X_test)[:, 1]
else:
    # fallback for models without predict_proba
    y_proba = None

print("\nClassification Report (class_weight='balanced'):")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print("Confusion Matrix:")
print(cm)
print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")

# Extra, helpful metrics for imbalanced data
if y_proba is not None:
    roc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)  # area under PR curve (good for imbalance)
    print(f"\nROC-AUC: {roc:.3f}")
    print(f"PR-AUC (Average Precision): {ap:.3f}")

# ---------- 7) (Optional) Lower the decision threshold to catch more quitters ----------
# By default, threshold is 0.5. Uncomment to try 0.35 and see recall increase:
"""
import numpy as np
thresh = 0.35
y_pred_t = (y_proba >= thresh).astype(int)
print(f"\n-- Threshold tuning at {thresh:.2f} --")
print(classification_report(y_test, y_pred_t))
print(confusion_matrix(y_test, y_pred_t))

"""
