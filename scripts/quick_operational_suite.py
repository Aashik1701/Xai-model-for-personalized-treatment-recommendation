#!/usr/bin/env python3
"""
Operational Model Suite - Quick Training Demo

A simplified version for quick testing and demonstration.
"""

import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.datasets import make_classification
import joblib
from datetime import datetime
import json

# Add paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from mlflow_manager import MLflowManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("quick_training")


class QuickModelSuite:
    """Quick operational model suite for testing."""

    def __init__(self):
        """Initialize the model suite."""
        self.mlflow_manager = MLflowManager(experiment_name="quick-operational-models")

        # Create output directory
        self.output_dir = Path("reports/quick_models")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Quick Model Suite initialized")

    def create_sample_data(self):
        """Create sample healthcare data."""
        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=7,
            n_redundant=2,
            n_classes=2,
            class_sep=0.8,
            random_state=42,
        )

        # Create feature names
        feature_names = [
            "age",
            "bmi",
            "blood_pressure",
            "cholesterol",
            "glucose",
            "heart_rate",
            "smoking",
            "exercise",
            "family_history",
            "stress_level",
        ]

        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name="cardiovascular_risk")

        return X_df, y_series

    def prepare_data(self, X, y):
        """Prepare training and test data."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )

        return {
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train,
            "y_test": y_test,
            "scaler": scaler,
        }

    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance."""
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics.update(
            {
                "precision": report["weighted avg"]["precision"],
                "recall": report["weighted avg"]["recall"],
                "f1_score": report["weighted avg"]["f1-score"],
            }
        )

        # Log to MLflow (simplified)
        with self.mlflow_manager.start_run(run_name=f"{model_name}_evaluation"):
            self.mlflow_manager.log_model_metrics(metrics, prefix=model_name)

        logger.info(
            f"{model_name} - ROC-AUC: {metrics['roc_auc']:.3f}, "
            f"Accuracy: {metrics['accuracy']:.3f}"
        )

        return metrics

    def train_baseline_models(self, data):
        """Train baseline models with simple configurations."""
        results = {}

        # Logistic Regression
        logger.info("Training Logistic Regression...")
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(data["X_train"], data["y_train"])
        lr_metrics = self.evaluate_model(
            lr_model, data["X_test"], data["y_test"], "logistic_regression"
        )
        results["logistic_regression"] = {"model": lr_model, "metrics": lr_metrics}

        # Random Forest
        logger.info("Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=10
        )
        rf_model.fit(data["X_train"], data["y_train"])
        rf_metrics = self.evaluate_model(
            rf_model, data["X_test"], data["y_test"], "random_forest"
        )
        results["random_forest"] = {"model": rf_model, "metrics": rf_metrics}

        return results

    def save_models(self, results):
        """Save trained models."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_paths = {}

        for model_name, result in results.items():
            if result["metrics"]["roc_auc"] > 0.7:  # Only save good models
                model_path = self.output_dir / f"{model_name}_{timestamp}.joblib"
                joblib.dump(result["model"], model_path)
                saved_paths[model_name] = str(model_path)
                logger.info(f"Saved {model_name} to {model_path}")

        return saved_paths

    def generate_report(self, results, saved_paths):
        """Generate performance report."""
        timestamp = datetime.now().isoformat()

        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]["metrics"]["roc_auc"])

        report = {
            "timestamp": timestamp,
            "summary": {
                "best_model": {
                    "name": best_model[0],
                    "roc_auc": best_model[1]["metrics"]["roc_auc"],
                    "accuracy": best_model[1]["metrics"]["accuracy"],
                },
                "models_meeting_threshold": len(
                    [r for r in results.values() if r["metrics"]["roc_auc"] > 0.7]
                ),
            },
            "detailed_results": {
                name: result["metrics"] for name, result in results.items()
            },
            "saved_models": saved_paths,
        }

        # Save report
        report_path = (
            self.output_dir / f"training_report_{timestamp.split('T')[0]}.json"
        )
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to {report_path}")

        # Print summary
        print("\n🏆 QUICK TRAINING RESULTS")
        print("=" * 40)
        print(f"Best Model: {best_model[0]}")
        print(f"ROC-AUC: {best_model[1]['metrics']['roc_auc']:.3f}")
        print(f"Accuracy: {best_model[1]['metrics']['accuracy']:.3f}")
        print(
            f"Models meeting threshold (>0.7 ROC-AUC): {report['summary']['models_meeting_threshold']}"
        )

        return report


def main():
    """Main execution."""
    logger.info("🚀 Starting Quick Operational Model Suite")

    # Initialize suite
    suite = QuickModelSuite()

    try:
        # Create sample data
        logger.info("📊 Creating sample healthcare data...")
        X, y = suite.create_sample_data()
        logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

        # Prepare data
        logger.info("🔧 Preparing data...")
        data = suite.prepare_data(X, y)

        # Train models
        logger.info("🏥 Training baseline models...")
        results = suite.train_baseline_models(data)

        # Save models
        logger.info("💾 Saving trained models...")
        saved_paths = suite.save_models(results)

        # Generate report
        logger.info("📋 Generating performance report...")
        report = suite.generate_report(results, saved_paths)

        logger.info("✅ Quick operational model suite complete!")

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
