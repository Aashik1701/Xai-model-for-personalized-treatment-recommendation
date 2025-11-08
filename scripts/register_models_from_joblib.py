"""
Bulk-register selected .joblib models from models/trained_real_datasets into MLflow
and transition them to Production under canonical names used by the API.

Canonical names (API):
  - BreastCancer
  - HeartDisease
  - SyntheaLongitudinal
  - ISICSkinCancer
  - DrugReviews

Usage:
    ./venv/bin/python scripts/register_models_from_joblib.py

Prereqs:
    - MLflow server running at http://127.0.0.1:5000
    - .joblib models present under models/trained_real_datasets
"""

from __future__ import annotations

import os
from pathlib import Path
import joblib
import mlflow
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models" / "trained_real_datasets"
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


# Map API model names to specific .joblib files to register
SELECTION = {
    # Breast Cancer Wisconsin
    "BreastCancer": MODELS_DIR / "breast_cancer_wisconsin_RandomForest.joblib",
    # Heart Disease (UCI variant)
    "HeartDisease": MODELS_DIR / "heart_disease_uci_RandomForest.joblib",
    # Synthea synthetic longitudinal
    "SyntheaLongitudinal": MODELS_DIR / "synthea_longitudinal_RandomForest.joblib",
    # ISIC Skin Cancer imaging
    "ISICSkinCancer": MODELS_DIR / "skin_cancer_imaging_RandomForest.joblib",
    # Drug Reviews sentiment/ratings
    "DrugReviews": MODELS_DIR / "drug_reviews_RandomForest.joblib",
}


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    exp_name = "real-datasets-import"
    mlflow.set_experiment(exp_name)

    for reg_name, path in SELECTION.items():
        if not path.exists():
            print(f"[SKIP] Missing file for {reg_name}: {path}")
            continue

        print(f"\n=== Registering {reg_name} from {path.name} ===")
        model = joblib.load(path)

        algo = path.stem.split("_")[-1] if "_" in path.stem else "unknown"
        dataset = path.stem.rsplit("_", 1)[0]

        with mlflow.start_run(run_name=f"import_{reg_name}") as run:
            run_id = run.info.run_id
            mlflow.log_param("source_file", str(path))
            mlflow.log_param("dataset", dataset)
            mlflow.log_param("algorithm", algo)

            # Log sklearn model and register under canonical name
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=reg_name,
            )

        # Find the newly created version for this run
        versions = client.search_model_versions(f"name='{reg_name}'")
        ver = next((v for v in versions if v.run_id == run_id), None)
        if ver is None:
            # Fallback to latest if run_id matching fails
            versions_sorted = sorted(versions, key=lambda v: int(v.version))
            ver = versions_sorted[-1] if versions_sorted else None

        if ver is None:
            print(f"[WARN] Could not resolve model version for {reg_name}")
            continue

        # Set helpful tags
        client.set_model_version_tag(reg_name, ver.version, "use_case", reg_name)
        client.set_model_version_tag(reg_name, ver.version, "algorithm", algo)

        # Promote to Production, archiving any existing Production version
        client.transition_model_version_stage(
            name=reg_name,
            version=ver.version,
            stage="Production",
            archive_existing_versions=True,
        )

        print(f"[OK] {reg_name} -> version {ver.version} transitioned to Production")

    print("\nAll selected models processed.")


if __name__ == "__main__":
    main()
