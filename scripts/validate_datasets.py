"""Lightweight dataset validator for normalized *raw* CSV datasets.

Checks performed (tabular focus initially):
  - Existence & readable CSV
  - Row / column counts (non-empty)
  - Missing value counts & percentage per column
  - Basic target distribution if a `target` or `diagnosis` column exists
  - Simple schema consistency: no all-empty columns, no duplicate columns
  - Optional threshold alerts for excessive missingness

Extendable: image / text / JSON modalities can hook in later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import argparse
import json
import sys
import pandas as pd  # type: ignore

RAW_DIR = Path("data/raw")

DEFAULT_MISSING_THRESHOLD = 0.4  # 40%


def list_candidate_csvs(patterns: List[str] | None = None) -> List[Path]:
    if not RAW_DIR.exists():
        return []
    files = sorted(RAW_DIR.glob("*.csv"))
    if patterns:
        lowered = [p.lower() for p in patterns]
        files = [f for f in files if any(p in f.name.lower() for p in lowered)]
    return files


def validate_csv(
    path: Path,
    missing_threshold: float,
    imbalance_ratio_warn: float | None = None,
) -> Dict[str, Any]:
    info: Dict[str, Any] = {"file": path.name, "status": "ok"}
    try:
        df = pd.read_csv(path)
    except Exception as e:  # noqa: BLE001
        info["status"] = "read_error"
        info["error"] = str(e)
        return info

    # Basic shape
    rows, cols = df.shape
    info["rows"] = rows
    info["cols"] = cols
    if rows == 0 or cols == 0:
        info["status"] = "empty"
        return info

    # Duplicate columns
    if len(df.columns) != len(set(df.columns)):
        info["duplicate_columns"] = True
        info["status"] = "schema_warning"

    # All-empty columns
    all_empty = [c for c in df.columns if df[c].isna().all()]
    if all_empty:
        info["all_empty_columns"] = all_empty
        info["status"] = "schema_warning"

    # Missingness per column
    missing_counts = df.isna().sum()
    missing_pct = (missing_counts / rows).round(4)
    info["missing_total"] = int(missing_counts.sum())
    info["missing_columns_over_threshold"] = {
        c: float(missing_pct[c])
        for c in df.columns
        if missing_pct[c] > missing_threshold
    }

    # Target distribution heuristics
    target_col = None
    for candidate in ("target", "diagnosis", "label", "class"):
        if candidate in df.columns:
            target_col = candidate
            break
    if target_col:
        val_counts = df[target_col].value_counts(dropna=False).to_dict()
        info["target_col"] = target_col
        info["target_distribution"] = {str(k): int(v) for k, v in val_counts.items()}
        if imbalance_ratio_warn and len(val_counts) >= 2:
            counts_sorted = sorted(val_counts.values(), reverse=True)
            majority = counts_sorted[0]
            minority = counts_sorted[-1]
            ratio = majority / max(minority, 1)
            info["imbalance_ratio"] = round(ratio, 4)
            if ratio >= imbalance_ratio_warn:
                info["imbalance_warning"] = True

    return info


def run(
    patterns: List[str] | None,
    missing_threshold: float,
    imbalance_ratio_warn: float | None = None,
) -> Dict[str, Any]:
    results = []
    files = list_candidate_csvs(patterns)
    if not files:
        return {"files": [], "summary": {"converted_files_found": 0}}

    for f in files:
        results.append(validate_csv(f, missing_threshold, imbalance_ratio_warn))

    summary = {
        "converted_files_found": len(files),
        "ok": sum(1 for r in results if r["status"] == "ok"),
        "schema_warning": sum(1 for r in results if r["status"] == "schema_warning"),
        "read_error": sum(1 for r in results if r["status"] == "read_error"),
        "empty": sum(1 for r in results if r["status"] == "empty"),
        "total_missing_values": sum(r.get("missing_total", 0) for r in results),
    }
    return {"files": results, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate normalized raw CSV datasets")
    parser.add_argument(
        "--filter",
        nargs="*",
        help=(
            "Optional substrings to filter which CSVs to validate " "(case-insensitive)"
        ),
    )
    parser.add_argument(
        "--missing-threshold",
        type=float,
        default=DEFAULT_MISSING_THRESHOLD,
        help=(
            "Flag columns whose missing percentage exceeds this "
            "fraction (default 0.4)"
        ),
    )
    parser.add_argument(
        "--imbalance-warn-ratio",
        type=float,
        default=None,
        help=(
            "If set, flag classification targets whose majority/minority "
            "ratio exceeds this value."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output (otherwise a human summary table is printed)",
    )
    args = parser.parse_args()

    report = run(args.filter, args.missing_threshold, args.imbalance_warn_ratio)

    if args.json:
        json.dump(report, sys.stdout, indent=2)
        print()
        return

    # Human summary
    summary = report["summary"]
    print("=== Dataset Validation Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("\nPer-file details:")
    for r in report["files"]:
        status = r["status"]
        base = (
            f" - {r['file']}: {status} " f"({r.get('rows', '?')}x{r.get('cols', '?')})"
        )
        print(base)
        if (
            "missing_columns_over_threshold" in r
            and r["missing_columns_over_threshold"]
        ):
            print(
                "    columns > threshold:",
                r["missing_columns_over_threshold"],
            )
        if "target_col" in r:
            print("    target:", r["target_col"], r.get("target_distribution"))
            if "imbalance_ratio" in r:
                msg = f"    class_imbalance_ratio: {r['imbalance_ratio']}"
                if r.get("imbalance_warning"):
                    msg += " (WARNING)"
                print(msg)
        if status != "ok":
            for extra in ("all_empty_columns", "duplicate_columns", "error"):
                if extra in r:
                    print(f"    {extra}: {r[extra]}")


if __name__ == "__main__":
    main()
