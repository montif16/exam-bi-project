from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, r2_score, silhouette_score
import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(exist_ok=True)

def train_classifier(df: pd.DataFrame, features: list[str], target: str):
    X = df[features]
    y = df[target]
    stratify = y if y.nunique() <= 10 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    pipe = Pipeline([
        ("scale", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    joblib.dump(pipe, MODELS_DIR / "classifier.joblib")
    return {"accuracy": acc}

def train_regressor(df: pd.DataFrame, features: list[str], target: str):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = Pipeline([
        ("scale", StandardScaler(with_mean=False)),
        ("reg", DecisionTreeRegressor(random_state=42))
    ])
    pipe.fit(X_train, y_train)
    r2 = r2_score(y_test, pipe.predict(X_test))
    joblib.dump(pipe, MODELS_DIR / "regressor.joblib")
    return {"r2": r2}

def train_cluster(df: pd.DataFrame, features: list[str], k: int = 3):
    X = df[features]
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    joblib.dump(km, MODELS_DIR / f"kmeans_k{k}.joblib")
    return {"silhouette": score}
