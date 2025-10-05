
"""
Single-source ETL demo (CSV -> Python data structure).

Tasks covered:
2) Functions to load/transform to Python structure (list[dict]).
3) Ingest by calling those functions.
4) Clean data (trim strings, type conversions, missing values).
5) Optional anonymisation (hash names/emails).
6) Visualisation (matplotlib).
"""

from __future__ import annotations
from typing import List, Dict, Iterable
import csv, hashlib, math
from datetime import datetime
import os

import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
DATA = os.path.join(os.path.dirname(HERE), "data", "people.csv")
VIS = os.path.join(os.path.dirname(HERE), "visuals")

def load_csv_to_records(path: str) -> List[Dict]:
    """Load a CSV file into a list of dicts (Python data structure)."""
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]
    return rows

def clean_records(rows: List[Dict]) -> List[Dict]:
    """Basic cleaning: trim text, convert types, handle missing data."""
    cleaned = []
    for r in rows:
        name = (r.get("name") or "").strip() or None
        email = (r.get("email") or "").strip().lower() or None

        # age: to int or None
        age_raw = (r.get("age") or "").strip()
        age = int(age_raw) if age_raw.isdigit() else None

        # salary: to float or None
        sal_raw = (r.get("salary") or "").replace(",", "").strip().lower()
        salary = None
        if sal_raw and sal_raw.replace(".","",1).isdigit():
            salary = float(sal_raw)
        # joined: to datetime.date or None
        joined_raw = (r.get("joined") or "").strip()
        try:
            joined = datetime.fromisoformat(joined_raw).date()
        except ValueError:
            joined = None

        cleaned.append({
            "name": name,
            "email": email,
            "age": age,
            "salary": salary,
            "joined": joined
        })
    return cleaned

def anonymise(rows: List[Dict], fields=("name","email")) -> List[Dict]:
    """Hash selected fields using SHA256 (deterministic, non-reversible)."""
    out = []
    for r in rows:
        nr = r.copy()
        for field in fields:
            val = nr.get(field)
            if val is not None:
                digest = hashlib.sha256(str(val).encode("utf-8")).hexdigest()[:16]
                nr[field] = f"anon_{digest}"
        out.append(nr)
    return out

def to_dataframe(rows: Iterable[Dict]) -> pd.DataFrame:
    """Helper to go from list[dict] -> DataFrame for exploration/plots."""
    return pd.DataFrame(list(rows))

def visualise(df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Age histogram
    plt.figure()
    df["age"].dropna().astype(int).plot(kind="hist", bins=10, title="Age distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "age_hist.png"))
    plt.close()

    # Salary histogram
    plt.figure()
    df["salary"].dropna().astype(float).plot(kind="hist", bins=10, title="Salary distribution")
    plt.xlabel("Salary")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "salary_hist.png"))
    plt.close()

    # Ages over join date (scatter-like using plot)
    if df["joined"].notna().any() and df["age"].notna().any():
        plt.figure()
        df2 = df.dropna(subset=["joined","age"]).sort_values("joined")
        plt.plot(df2["joined"], df2["age"], marker="o")
        plt.title("Age over join date")
        plt.xlabel("Joined")
        plt.ylabel("Age")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "age_over_time.png"))
        plt.close()

def run_pipeline(csv_path: str = DATA, out_dir: str = VIS, do_anon: bool = False):
    # 2) Load/transform
    raw = load_csv_to_records(csv_path)

    # 4) Clean
    cleaned = clean_records(raw)

    # 5) Optional anonymisation
    rows = anonymise(cleaned) if do_anon else cleaned

    # 6) Visualisation
    df = to_dataframe(rows)
    visualise(df, out_dir)

    # Return core Python data structure so the caller can inspect/use it
    return rows

if __name__ == "__main__":
    rows = run_pipeline(do_anon=True)
    # print short preview to console
    import json
    print(json.dumps(rows[:3], default=str, indent=2))
