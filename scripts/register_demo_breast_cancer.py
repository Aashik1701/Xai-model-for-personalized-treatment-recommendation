"""
Register a demo BreastCancer model into MLflow and set it to Production.

This script:
- Trains a simple sklearn Pipeline (StandardScaler + LogisticRegression)
- Logs metrics and the model to MLflow
- Registers the model as "BreastCancer" and transitions it to Production
- Logs a background artifact for explainability (background/background.npy)
- Writes a ready-to-use sample prediction JSON to tests/sample_requests/

Run:
    venv/bin/python scripts/register_demo_breast_cancer.py

Requirements: scikit-learn, pandas, numpy, mlflow
MLflow UI expected at http://127.0.0.1:5000
"""

from __future__ import annotations

import os
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

import mlflow
from mlflow.tracking import MlflowClient


def main():
    project_root = Path(__file__).resolve().parents[1]

    # Ensure MLflow tracking URI matches the running server
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)

    # Data
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model pipeline
    pipe = Pipeline(
        steps=[
            ("preprocessor", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, solver="lbfgs")),
        ]
    )

    # Train
    pipe.fit(X_train, y_train)

    # Evaluate
    y_pred = pipe.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))

    # Log to MLflow
    mlflow.set_experiment("demo-breast-cancer")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("pipeline", "StandardScaler+LogReg")
        mlflow.log_metric("accuracy", acc)

        # Background artifact for explainability (original feature space)
        bg = X_train.copy().to_numpy()
        # Subsample to reasonable size
        if bg.shape[0] > 512:
            rng = np.random.default_rng(123)
            idx = rng.choice(bg.shape[0], size=512, replace=False)
            bg = bg[idx]

        with tempfile.TemporaryDirectory() as tmpdir:
            bg_dir = Path(tmpdir) / "background"
            bg_dir.mkdir(parents=True, exist_ok=True)
            bg_path = bg_dir / "background.npy"
            np.save(bg_path, bg)
            mlflow.log_artifacts(str(bg_dir), artifact_path="background")

        # Input example and model logging/registration
        input_example = X_test.iloc[[0]]
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            registered_model_name="BreastCancer",
            input_example=input_example,
        )

    # Post-registration: set tags and transition to Production
    client = MlflowClient()
    versions = client.search_model_versions("name='BreastCancer'")
    # Find the version associated with our run
    version_obj = next((v for v in versions if v.run_id == run_id), None)
    if version_obj is None:
        raise RuntimeError("Failed to find registered model version for the run")

    # Set some tags for UI listing
    client.set_model_version_tag(
        "BreastCancer", version_obj.version, "accuracy", f"{acc:.4f}"
    )
    client.set_model_version_tag(
        "BreastCancer", version_obj.version, "use_case", "Breast cancer diagnosis"
    )

    # Transition to Production
    client.transition_model_version_stage(
        name="BreastCancer",
        version=version_obj.version,
        stage="Production",
        archive_existing_versions=True,
    )

    # Write a ready-to-use sample predict JSON matching the model's features
    sample_dir = project_root / "tests" / "sample_requests"
    sample_dir.mkdir(parents=True, exist_ok=True)

    sample_features = {
        str(col): float(input_example.iloc[0][col]) for col in input_example.columns
    }
    sample_request = {
        "model_name": "BreastCancer",
        "patient_data": {"features": sample_features},
        "return_probabilities": True,
        "include_explanation": False,
        "include_personalization": False,
    }
    out_path = sample_dir / "breast_cancer_predict.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(sample_request, f, indent=2)

    print("\n====================================================")
    print("Registered BreastCancer model in MLflow and set to Production")
    print(f"Tracking URI: {tracking_uri}")
    print(f"Model version: {version_obj.version}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Sample request written to: {out_path}")
    print("Now you can call the API /predict with this file.")
    print("====================================================\n")


if __name__ == "__main__":
    main()
