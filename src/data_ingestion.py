from pathlib import Path
import pandas as pd
import csv

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW = DATA_DIR / "raw"
PROCESSED = DATA_DIR / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

def _read_smart(path: Path) -> pd.DataFrame:
    # If extension is Excel or file header is PK.. (xlsx/zip), read as Excel
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    # XLSX disguised as .csv? Check header
    head = path.read_bytes()[:4]
    if head.startswith(b"PK\x03\x04"):
        return pd.read_excel(path)

    # Otherwise, detect delimiter and read as text CSV
    # Try encodings + sniff delimiters
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            sample = path.open("r", encoding=enc, errors="strict").read(65536)
            try:
                delim = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
            except Exception:
                if sample.count(";") > sample.count(","):
                    delim = ";"
                elif "\t" in sample:
                    delim = "\t"
                else:
                    delim = ","
            return pd.read_csv(path, sep=delim, encoding=enc, engine="python")
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError:
            pass
    # last resort
    return pd.read_csv(path, sep=";", encoding="latin-1", engine="python", on_bad_lines="skip")

def build_dataset() -> pd.DataFrame:
    # Prefer xlsx if present; else fall back to dataset.csv
    xlsx = RAW / "dataset.xlsx"
    csvp = RAW / "dataset.csv"
    src = xlsx if xlsx.exists() else csvp
    if not src.exists():
        raise FileNotFoundError(f"Place your file at {xlsx} or {csvp}")

    df_raw = _read_smart(src)

    if "G3" not in df_raw.columns:
        raise ValueError(f"'G3' not in columns: {list(df_raw.columns)[:12]} ...")

    # Coerce common numeric cols
    for col in ["G1","G2","G3","age","studytime","failures","absences"]:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

    # Binary label for classification
    df_raw["Pass"] = (df_raw["G3"] >= 10).astype("Int64")

    # One-hot encode categoricals
    df_proc = pd.get_dummies(df_raw, columns=[c for c in df_raw.columns if df_raw[c].dtype == "object"],
                             drop_first=True, dtype=int)

    # Drop rows missing targets
    df_proc = df_proc.dropna(subset=["G3","Pass"]).reset_index(drop=True)

    out = PROCESSED / "dataset_clean.csv"
    df_proc.to_csv(out, index=False)
    return df_proc
