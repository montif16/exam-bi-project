import pandas as pd
import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "6"  
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------- 1) Load and clean ----------
df = pd.read_csv("data/hr.csv", encoding="utf-8-sig")
df = df.drop(columns=["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"], errors="ignore")

# Features (drop target columns)
X = df.drop(columns=["Attrition", "MonthlyIncome"])

# Split numeric vs categorical
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

# ---------- 2) Preprocessing ----------
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

# ---------- 3) Try clustering with different k ----------
results = {}
for k in range(2, 11):  # try 2–10 clusters
    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("cluster", KMeans(n_clusters=k, random_state=42, n_init=10))
    ])
    pipe.fit(X)
    labels = pipe.named_steps["cluster"].labels_
    score = silhouette_score(pipe.named_steps["preprocess"].transform(X), labels)
    results[k] = score
    print(f"k={k}: silhouette={score:.3f}")

# ---------- 4) Pick best k ----------
best_k = max(results, key=results.get)
print(f"\nBest number of clusters: k={best_k} with silhouette={results[best_k]:.3f}")

# ---------- 5) Refit best model ----------
best_pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("cluster", KMeans(n_clusters=best_k, random_state=42, n_init=10))
])
best_pipe.fit(X)
df["Cluster"] = best_pipe.named_steps["cluster"].labels_

print("\nCluster distribution:")
print(df["Cluster"].value_counts())
