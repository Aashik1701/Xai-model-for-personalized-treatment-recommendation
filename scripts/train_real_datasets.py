"""
Real Dataset Training and Explainability Testing

Train models on real healthcare datasets and test explainability toolkit.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model_on_dataset(dataset_name: str, model_type: str = "random_forest"):
    """Train a model on a real dataset and save it for explainability testing."""

    # Load dataset
    data_path = Path(f"data/processed/{dataset_name}_processed.csv")
    if not data_path.exists():
        logger.error(f"Dataset not found: {data_path}")
        return None

    logger.info(f"🏥 Training {model_type} on {dataset_name}")

    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Dataset shape: {df.shape}")

    # Prepare features and target
    if "target" not in df.columns:
        logger.error("No 'target' column found in dataset")
        return None

    # Remove non-feature columns
    feature_cols = [col for col in df.columns if col not in ["target", "patient_id"]]
    X = df[feature_cols]
    y = df["target"]

    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame for compatibility
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Train model
    if model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
    elif model_type == "logistic_regression":
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)
    else:
        logger.error(f"Unknown model type: {model_type}")
        return None

    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    logger.info(f"✅ Model trained successfully!")
    logger.info(f"Accuracy: {accuracy:.3f}")
    logger.info(f"Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model and data
    model_dir = Path("reports/real_dataset_models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{dataset_name}_{model_type}.joblib"
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "feature_names": feature_cols,
            "X_test": X_test_scaled,
            "y_test": y_test,
            "accuracy": accuracy,
            "dataset_name": dataset_name,
        },
        model_path,
    )

    logger.info(f"💾 Model saved: {model_path}")
    return model_path


def main():
    """Train models on multiple real datasets."""

    # Available datasets
    datasets = [
        "heart_disease_uci",
        "breast_cancer_wisconsin",
        "diabetes_130_hospitals",
        "hepatitis",
        "dermatology",
    ]

    # Train models
    trained_models = []

    for dataset in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"TRAINING ON {dataset.upper()}")
        logger.info(f"{'='*60}")

        # Train Random Forest
        rf_path = train_model_on_dataset(dataset, "random_forest")
        if rf_path:
            trained_models.append(rf_path)

        # Train Logistic Regression (for smaller datasets)
        if dataset in ["heart_disease_uci", "breast_cancer_wisconsin", "hepatitis"]:
            lr_path = train_model_on_dataset(dataset, "logistic_regression")
            if lr_path:
                trained_models.append(lr_path)

    # Summary
    logger.info(f"\n🎉 TRAINING COMPLETE!")
    logger.info(f"Trained {len(trained_models)} models:")
    for model_path in trained_models:
        logger.info(f"  - {model_path}")

    logger.info(f"\n💡 Next steps:")
    logger.info(
        f"  1. Test explainability: python scripts/simple_explainer.py --model-path <model_path>"
    )
    logger.info(
        f"  2. Test personalization: python scripts/personalization_cli.py --data <dataset_path>"
    )

    return trained_models


if __name__ == "__main__":
    main()
