"""Converters for additional healthcare datasets (stubs + partial logic).

Datasets targeted:
  - Diabetes 130 US hospitals (reduced feature subset)
  - Thyroid disease (UCI) consolidation
  - Dermatology (UCI)

Each converter should output a normalized CSV into data/raw/ matching paths in
config/data_config.yaml. This file is incremental: fill in TODOs once raw source
archives are added under data/ (e.g., data/thyroid/, data/dermatology/ etc.).
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd  # type: ignore
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("convert_additional")

DATA_DIR = Path("data")
RAW_OUT = DATA_DIR / "raw"
RAW_OUT.mkdir(parents=True, exist_ok=True)


def convert_diabetes_130() -> Path | None:
    """Convert diabetes 130 US hospitals dataset.

    Expectation: CSV at data/external/diabetes_130_hospitals.csv.
    Light column pruning + target normalization (readmitted -> binary).
    """
    src = DATA_DIR / "external" / "diabetes_130_hospitals.csv"
    if not src.exists():
        logger.warning("Diabetes source not found: %s", src)
        return None
    df = pd.read_csv(src)
    # Typical high-cardinality ID columns to drop if present
    drop_cols = [c for c in ["encounter_id", "patient_nbr"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    # Normalize readmitted target to binary: <30 or >30 -> 1, NO -> 0
    if "readmitted" in df.columns:
        df["readmitted"] = df["readmitted"].apply(lambda v: 0 if v == "NO" else 1)
    out = RAW_OUT / "diabetes_130_hospitals.csv"
    df.to_csv(out, index=False)
    logger.info(
        "Diabetes 130 -> %s (rows=%d, cols=%d)",
        out,
        len(df),
        df.shape[1],
    )
    return out


def convert_thyroid() -> Path | None:
    """Convert thyroid dataset (placeholder).

    TODO: Implement once raw files are added. Usually multiple TSV/CSV parts.
    Steps will include:
      - Merge training/test parts
      - Map class labels to target (e.g., hyper / hypo / normal) or binary
      - Handle '?' missing markers
    """
    src_dir = DATA_DIR / "thyroid"
    if not src_dir.exists():
        logger.warning("Thyroid source directory missing: %s", src_dir)
        return None
    # Expect files: thyroid_train.csv / thyroid_test.csv OR a single file
    candidates = list(src_dir.glob("*.csv")) + list(src_dir.glob("*.data"))
    if not candidates:
        logger.warning("No thyroid source files found in %s", src_dir)
        return None
    frames = []
    for p in candidates:
        try:
            df_part = pd.read_csv(p, na_values=["?", "NA", "nan"], header=None)
            frames.append(df_part)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed reading %s: %s", p, e)
    if not frames:
        return None
    df = pd.concat(frames, axis=0, ignore_index=True)
    # Heuristic: last column is label
    if df.shape[1] < 2:
        logger.warning("Thyroid dataset shape unexpected: %s", df.shape)
        return None
    feature_cols = [f"feature_{i}" for i in range(1, df.shape[1])]
    col_names = feature_cols + ["target_raw"]
    df.columns = col_names
    # Map raw target labels to simplified categories if common patterns found
    if df["target_raw"].dtype == "object":
        mapping = {}
        unique_labels = set(str(v).lower() for v in df["target_raw"].unique())
        # Basic pattern grouping
        for lbl in unique_labels:
            if "hyper" in lbl:
                mapping[lbl] = "hyper"
            elif "hypo" in lbl:
                mapping[lbl] = "hypo"
            elif "negative" in lbl or "normal" in lbl:
                mapping[lbl] = "normal"
            else:
                mapping[lbl] = lbl
        df["target"] = [mapping[str(v).lower()] for v in df["target_raw"]]
    else:
        df["target"] = df["target_raw"]
    df.drop(columns=["target_raw"], inplace=True)
    out = RAW_OUT / "thyroid.csv"
    df.to_csv(out, index=False)
    logger.info(
        "Thyroid -> %s (rows=%d, cols=%d, missing=%d)",
        out,
        len(df),
        df.shape[1],
        int(df.isna().sum().sum()),
    )
    return out


def convert_dermatology() -> Path | None:
    """Convert dermatology dataset (UCI)."""
    src = DATA_DIR / "dermatology" / "dermatology.data"
    if not src.exists():
        logger.warning("Dermatology source not found: %s", src)
        return None
    # UCI: 34 ordinal + age + family history + target
    # Missing values represented by '?' in age.
    cols = [f"feature_{i}" for i in range(1, 35)] + [
        "age",
        "family_history",
        "target",
    ]
    df = pd.read_csv(src, header=None, names=cols, na_values=["?"])
    out = RAW_OUT / "dermatology.csv"
    df.to_csv(out, index=False)
    logger.info(
        "Dermatology -> %s (rows=%d, cols=%d, missing=%d)",
        out,
        len(df),
        df.shape[1],
        int(df.isna().sum().sum()),
    )
    return out


def main() -> None:
    produced = []
    for fn in (convert_diabetes_130, convert_thyroid, convert_dermatology):
        out = fn()
        if out:
            produced.append(out.name)
    if produced:
        logger.info("Converted: %s", ", ".join(produced))
    else:
        logger.info("No additional datasets converted (check sources).")


if __name__ == "__main__":
    main()
