#!/usr/bin/env python3
"""
Quick test of MLflow experiment tracking functionality.

This script demonstrates logging a simple training run with metrics and model artifacts.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import mlflow

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mlflow_manager import MLflowManager


def create_sample_data():
    """Create sample classification data for testing."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42,
    )

    # Convert to DataFrame for consistency with real datasets
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")

    return train_test_split(X_df, y_series, test_size=0.2, random_state=42)


def test_mlflow_tracking():
    """Test MLflow experiment tracking with a sample model."""
    print("🧪 Testing MLflow Experiment Tracking...")

    # Initialize MLflow manager
    mlflow_manager = MLflowManager(experiment_name="test-healthcare-models")

    # Create sample data
    X_train, X_test, y_train, y_test = create_sample_data()

    # Start MLflow run
    with mlflow_manager.start_run(
        run_name="test_random_forest",
        tags={"test": "true", "model_type": "random_forest"},
    ):
        # Log dataset information
        dataset_config = {
            "raw_path": "synthetic_test_data",
            "target_column": "target",
            "n_samples": len(X_train) + len(X_test),
            "n_features": X_train.shape[1],
        }
        mlflow_manager.log_dataset_info("synthetic_classification", dataset_config)

        # Log training parameters
        mlflow_manager.log_training_params(
            n_estimators=100, max_depth=10, random_state=42, class_weight="balanced"
        )

        # Train model
        model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, class_weight="balanced"
        )
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        # Log metrics
        metrics = {
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
        }
        mlflow_manager.log_model_metrics(metrics)

        # Log model
        model_uri = mlflow_manager.log_ensemble_model(
            model=model,
            model_name="test_random_forest",
            input_example=X_test.head(1).to_dict("records")[0],
        )

        # Register model if performance is good
        if roc_auc > 0.8:
            mlflow_manager.register_model(
                model_uri=model_uri,
                model_name="test_healthcare_model",
                stage="Staging",
                description=f"Test Random Forest model with ROC-AUC: {roc_auc:.3f}",
            )

        print(f"✅ Run completed successfully!")
        print(f"   - Accuracy: {accuracy:.3f}")
        print(f"   - ROC-AUC: {roc_auc:.3f}")
        print(f"   - Model URI: {model_uri}")
        print(f"   - Run ID: {mlflow.active_run().info.run_id}")


def test_model_loading():
    """Test loading models from the registry."""
    print("\n🔄 Testing Model Loading from Registry...")

    mlflow_manager = MLflowManager(experiment_name="test-healthcare-models")

    # Try to load the model we just registered
    model = mlflow_manager.get_best_model("test_healthcare_model", stage="Staging")

    if model is not None:
        print("✅ Model loaded successfully from registry!")

        # Test prediction with the loaded model
        X_train, X_test, y_train, y_test = create_sample_data()
        predictions = model.predict(X_test[:5])  # Test with first 5 samples
        print(f"   - Sample predictions: {predictions}")
    else:
        print("⚠️  No model found in registry (this is expected for first run)")


if __name__ == "__main__":
    print("🏥 MLflow Healthcare Experiment Tracking Test")
    print("=" * 50)

    try:
        test_mlflow_tracking()
        test_model_loading()

        print("\n🎉 All tests completed successfully!")
        print("\n📊 MLflow UI is available at: http://localhost:5000")
        print("   Visit the UI to see the logged experiments, metrics, and models!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
