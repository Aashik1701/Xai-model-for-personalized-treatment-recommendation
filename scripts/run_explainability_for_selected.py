"""
Run ExplainabilityToolkit on selected trained models and log results to MLflow
experiment "explainability-analysis" (metrics + local reports/plots).

Usage:
    ./venv/bin/python scripts/run_explainability_for_selected.py

Requires:
    - MLflow server running (http://127.0.0.1:5000)
    - SHAP/LIME installed in the project venv
    - Trained .joblib models under models/trained_real_datasets
"""

from __future__ import annotations

from pathlib import Path
import sys
from explainability_toolkit import ExplainabilityToolkit  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def main():
    models_dir = PROJECT_ROOT / "models" / "trained_real_datasets"

    selection = {
        "BreastCancer": models_dir / "breast_cancer_wisconsin_RandomForest.joblib",
        "HeartDisease": models_dir / "heart_disease_uci_RandomForest.joblib",
        "SyntheaLongitudinal": models_dir / "synthea_longitudinal_RandomForest.joblib",
        "ISICSkinCancer": models_dir / "skin_cancer_imaging_RandomForest.joblib",
        "DrugReviews": models_dir / "drug_reviews_RandomForest.joblib",
    }

    toolkit = ExplainabilityToolkit("config/explainability_config.yaml")

    for name, path in selection.items():
        if not path.exists():
            print(f"[SKIP] Missing model file for {name}: {path}")
            continue
        print(f"\n=== Generating explanations for {name} ===")
        try:
            toolkit.explain_model(str(path), model_name=name)
            print(
                f"[OK] {name} explanations logged to MLflow experiment "
                f"'explainability-analysis'"
            )
        except Exception as e:
            print(f"[FAIL] {name}: {e}")

    print("\nAll selected models processed for explainability.")


if __name__ == "__main__":
    main()
