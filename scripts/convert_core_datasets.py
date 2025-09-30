"""Convert core small tabular medical datasets to normalized CSV form.

Datasets handled:
  - Heart Disease (UCI) processed Cleveland subset
  - Breast Cancer Wisconsin (Diagnostic WDBC)
  - Hepatitis (UCI)

Outputs written under data/raw/ with standardized snake_case names.
Repeatable & idempotent: existing outputs will be overwritten.
"""

from pathlib import Path
from typing import Optional, List
import pandas as pd  # type: ignore
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("convert_core_datasets")

DATA_DIR = Path("data")
RAW_OUT = DATA_DIR / "raw"
RAW_OUT.mkdir(parents=True, exist_ok=True)


def _coerce_numeric(df: pd.DataFrame, exclude: List[str]) -> None:
    """In-place numeric coercion for columns not in exclude."""
    for col in df.columns:
        if col not in exclude:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def convert_heart_disease() -> Optional[Path]:
    src = DATA_DIR / "heart+disease" / "processed.cleveland.data"
    if not src.exists():
        logger.warning("Heart disease source not found: %s", src)
        return None
    cols = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "target",
    ]
    df = pd.read_csv(src, names=cols, na_values=["?"])
    # Basic type coercion
    categorical_like = (
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
        "target",
    )
    numeric_cols = [c for c in df.columns if c not in categorical_like]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    out = RAW_OUT / "heart_disease.csv"
    df.to_csv(out, index=False)
    missing_total = int(df.isna().sum().sum())
    logger.info(
        "Heart Disease -> %s (rows=%d, missing=%d)",
        out,
        len(df),
        missing_total,
    )
    return out


def convert_breast_cancer() -> Optional[Path]:
    src = DATA_DIR / "breast+cancer+wisconsin+diagnostic" / "wdbc.data"
    if not src.exists():
        logger.warning("Breast cancer source not found: %s", src)
        return None
    cols = ["id", "diagnosis"] + [f"feature_{i}" for i in range(1, 31)]
    df = pd.read_csv(src, header=None, names=cols)
    # Drop id if not needed downstream; keep for traceability for now
    out = RAW_OUT / "breast_cancer_wisconsin.csv"
    df.to_csv(out, index=False)
    logger.info("Breast Cancer -> %s (rows=%d)", out, len(df))
    return out


def convert_hepatitis() -> Optional[Path]:
    src = DATA_DIR / "hepatitis" / "hepatitis.data"
    if not src.exists():
        logger.warning("Hepatitis source not found: %s", src)
        return None
    # UCI Hepatitis has 20 columns inc. class label first
    cols = [
        "target",
        "age",
        "sex",
        "steroid",
        "antivirals",
        "fatigue",
        "malaise",
        "anorexia",
        "liver_big",
        "liver_firm",
        "spleen_palpable",
        "spiders",
        "ascites",
        "varices",
        "bilirubin",
        "alk_phosphate",
        "sgot",
        "albumin",
        "protime",
        "histology",
    ]
    df = pd.read_csv(src, header=None, names=cols, na_values=["?"])
    out = RAW_OUT / "hepatitis.csv"
    df.to_csv(out, index=False)
    logger.info(
        "Hepatitis -> %s (rows=%d, missing=%d)",
        out,
        len(df),
        int(df.isna().sum().sum()),
    )
    return out


def main() -> None:
    logger.info("Starting core dataset conversions...")
    produced = []
    for fn in (
        convert_heart_disease,
        convert_breast_cancer,
        convert_hepatitis,
    ):
        out = fn()
        if out:
            produced.append(out)
    if produced:
        joined = ", ".join(str(p.name) for p in produced)
        logger.info("Completed conversions: %s", joined)
        return
    logger.warning("No datasets converted. Check source paths.")


if __name__ == "__main__":
    main()
