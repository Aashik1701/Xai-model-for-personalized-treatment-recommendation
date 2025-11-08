"""
Wrap joblib sklearn models as MLflow pyfunc models and register them.
This ensures the 'python_function' flavor is present so the API's mlflow.pyfunc.load_model works.

Usage:
    ./venv/bin/python scripts/register_joblib_as_pyfunc.py
"""

from pathlib import Path
import os
import joblib
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.pyfunc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models" / "trained_real_datasets"
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

SELECTION = {
    "BreastCancer": MODELS_DIR / "breast_cancer_wisconsin_RandomForest.joblib",
    "HeartDisease": MODELS_DIR / "heart_disease_uci_RandomForest.joblib",
    "SyntheaLongitudinal": MODELS_DIR / "synthea_longitudinal_RandomForest.joblib",
    "ISICSkinCancer": MODELS_DIR / "skin_cancer_imaging_RandomForest.joblib",
    "DrugReviews": MODELS_DIR / "drug_reviews_RandomForest.joblib",
}


class SklearnJoblibWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import joblib

        path = context.artifacts.get("joblib_model")
        if path is None:
            raise RuntimeError("No joblib_model artifact found in context")

        # Load the artifact (may be a dict wrapper or raw estimator)
        import os

        # If MLflow passed a directory, try to find a joblib/pkl file inside
        if os.path.isdir(path):
            candidates = [
                p
                for p in [os.path.join(path, f) for f in os.listdir(path)]
                if os.path.isfile(p) and (p.endswith(".joblib") or p.endswith(".pkl"))
            ]
            if candidates:
                path = candidates[0]

        loaded = joblib.load(path)

        # Try to find an sklearn-like estimator (has predict) inside nested structures
        def _find_estimator(obj):
            # Direct estimator
            if hasattr(obj, "predict"):
                return obj
            # dict-like wrapper
            if isinstance(obj, dict):
                # common key names
                for key in ("model", "estimator", "sklearn_model", "clf"):
                    if key in obj and hasattr(obj[key], "predict"):
                        return obj[key]
                # search values recursively
                for v in obj.values():
                    found = _find_estimator(v)
                    if found is not None:
                        return found
            # list/tuple - search items
            if isinstance(obj, (list, tuple)):
                for item in obj:
                    found = _find_estimator(item)
                    if found is not None:
                        return found
            return None

        est = _find_estimator(loaded)
        if est is None:
            # fallback: keep loaded object and hope it implements predict
            self.model = loaded
        else:
            self.model = est

    def predict(self, context, model_input):
        # model_input is a pandas DataFrame
        try:
            preds = self.model.predict(model_input)
            return preds
        except Exception:
            # fallback: if model expects numpy arrays
            return self.model.predict(model_input.values)


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    mlflow.set_experiment("pyfunc-import")

    for name, joblib_path in SELECTION.items():
        if not joblib_path.exists():
            print(f"[SKIP] Missing {joblib_path}")
            continue

        print(f"Registering {name} as pyfunc from {joblib_path}")
        with mlflow.start_run(run_name=f"pyfunc_import_{name}") as run:
            run_id = run.info.run_id
            # Log a small param
            mlflow.log_param("source_file", str(joblib_path))

            # Prepare artifacts map
            artifacts = {"joblib_model": str(joblib_path)}

            # Log the pyfunc model wrapper
            mlflow.pyfunc.log_model(
                python_model=SklearnJoblibWrapper(),
                artifacts=artifacts,
                registered_model_name=name,
                artifact_path="model",
            )

        # Transition to Production
        versions = client.search_model_versions(f"name='{name}'")
        ver = next((v for v in versions if v.run_id == run_id), None)
        if ver is None:
            versions_sorted = sorted(versions, key=lambda v: int(v.version))
            ver = versions_sorted[-1] if versions_sorted else None
        if ver is None:
            print(f"[WARN] Could not resolve version for {name}")
            continue

        client.transition_model_version_stage(
            name=name,
            version=ver.version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"[OK] Registered {name} as pyfunc -> version {ver.version} (Production)")


if __name__ == "__main__":
    main()
