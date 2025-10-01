#!/usr/bin/env python3
"""
Enhanced Training Pipeline with Operational Model Suite Integration

This script provides comprehensive model training with:
- Automated hyperparameter optimization
- Performance benchmarking
- Fairness assessment
- Production-ready model validation
"""

import sys
from pathlib import Path
import logging
import argparse
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from datetime import datetime

# Add project paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mlflow_manager import MLflowManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("enhanced_training")


class EnhancedTrainingPipeline:
    """Enhanced training pipeline with comprehensive evaluation."""

    def __init__(self, config_path: str = None):
        """Initialize the training pipeline."""
        self.config_path = config_path or "config/operational_model_config.yaml"
        self.config = self._load_config()

        # Initialize MLflow
        self.mlflow_manager = MLflowManager(
            experiment_name=self.config.get("mlflow", {}).get(
                "experiment_name", "enhanced-training"
            )
        )

        # Create output directories
        self.output_dir = Path("models/trained")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Enhanced Training Pipeline initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "baseline_models": {
                "logistic_regression": {
                    "class": "LogisticRegression",
                    "hyperparameters": {"C": [0.1, 1.0, 10.0]},
                },
                "random_forest": {
                    "class": "RandomForestClassifier",
                    "hyperparameters": {"n_estimators": [50, 100]},
                },
            },
            "evaluation": {
                "cross_validation": {"folds": 5},
                "test_split": {"test_size": 0.2, "random_state": 42},
            },
            "thresholds": {"minimum_roc_auc": 0.7},
        }

    def prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Prepare data for training with preprocessing."""
        logger.info("Preparing data for training")

        # Split data
        eval_config = self.config.get("evaluation", {})
        split_config = eval_config.get("test_split", {})

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=split_config.get("test_size", 0.2),
            random_state=split_config.get("random_state", 42),
            stratify=y if split_config.get("stratify", True) else None,
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
            "original_X_train": X_train,
            "original_X_test": X_test,
        }

    def create_model(self, model_name: str, model_config: Dict[str, Any]):
        """Create model instance from configuration."""
        model_class = model_config.get("class")

        model_mapping = {
            "LogisticRegression": LogisticRegression(random_state=42),
            "RandomForestClassifier": RandomForestClassifier(random_state=42),
            "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
            "SVC": SVC(random_state=42, probability=True),
        }

        if model_class not in model_mapping:
            raise ValueError(f"Unknown model class: {model_class}")

        return model_mapping[model_class]

    def optimize_hyperparameters(
        self,
        model,
        param_grid: Dict[str, List],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_name: str,
    ) -> Any:
        """Optimize hyperparameters using GridSearchCV."""
        logger.info(f"Optimizing hyperparameters for {model_name}")

        # Create pipeline with scaling
        pipeline = Pipeline([("model", model)])

        # Adjust parameter names for pipeline
        pipeline_params = {f"model__{k}": v for k, v in param_grid.items()}

        # Grid search
        grid_search = GridSearchCV(
            pipeline,
            pipeline_params,
            cv=self.config.get("evaluation", {})
            .get("cross_validation", {})
            .get("folds", 5),
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.3f}")

        return grid_search.best_estimator_

    def evaluate_model(
        self, model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str
    ) -> Dict[str, Any]:
        """Evaluate model performance."""
        logger.info(f"Evaluating {model_name}")

        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        roc_auc = roc_auc_score(y_test, y_proba)
        report = classification_report(y_test, y_pred, output_dict=True)

        metrics = {
            "roc_auc": roc_auc,
            "accuracy": report["accuracy"],
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1_score": report["weighted avg"]["f1-score"],
        }

        # Log to MLflow
        self.mlflow_manager.log_metrics(metrics)

        # Check thresholds
        min_roc_auc = self.config.get("thresholds", {}).get("minimum_roc_auc", 0.0)
        if roc_auc >= min_roc_auc:
            logger.info(f"✅ {model_name} meets performance thresholds")
            metrics["meets_threshold"] = True
        else:
            logger.warning(
                f"⚠️ {model_name} below threshold "
                f"(ROC-AUC: {roc_auc:.3f} < {min_roc_auc})"
            )
            metrics["meets_threshold"] = False

        return metrics

    def save_model(self, model, model_name: str, metrics: Dict[str, Any]):
        """Save trained model with metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name}_{timestamp}.joblib"
        model_path = self.output_dir / model_filename

        # Save model
        joblib.dump(model, model_path)

        # Save metadata
        metadata = {
            "model_name": model_name,
            "timestamp": timestamp,
            "metrics": metrics,
            "model_path": str(model_path),
            "config": self.config,
        }

        metadata_path = self.output_dir / f"{model_name}_{timestamp}_metadata.json"
        import json

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Model saved: {model_path}")
        logger.info(f"Metadata saved: {metadata_path}")

        return model_path

    def train_baseline_models(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Train all baseline models."""
        logger.info("🏥 Training baseline models")
        results = {}

        baseline_config = self.config.get("baseline_models", {})

        for model_name, model_config in baseline_config.items():
            logger.info(f"Training {model_name}")

            try:
                # Create model
                model = self.create_model(model_name, model_config)

                # Optimize hyperparameters if specified
                hyperparams = model_config.get("hyperparameters", {})
                if hyperparams:
                    model = self.optimize_hyperparameters(
                        model, hyperparams, data["X_train"], data["y_train"], model_name
                    )
                else:
                    # Simple training
                    model.fit(data["X_train"], data["y_train"])

                # Evaluate
                metrics = self.evaluate_model(
                    model, data["X_test"], data["y_test"], model_name
                )

                # Save if meets thresholds
                if metrics.get("meets_threshold", False):
                    model_path = self.save_model(model, model_name, metrics)
                    metrics["model_path"] = str(model_path)

                results[model_name] = {"model": model, "metrics": metrics}

                logger.info(f"✅ {model_name} training complete")

            except Exception as e:
                logger.error(f"❌ Failed to train {model_name}: {e}")
                results[model_name] = {"error": str(e)}

        return results

    def create_ensemble_models(
        self, base_results: Dict[str, Any], data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create and train ensemble models."""
        logger.info("🔗 Creating ensemble models")

        # Filter successful base models
        successful_models = {
            name: result["model"]
            for name, result in base_results.items()
            if "model" in result
            and result.get("metrics", {}).get("meets_threshold", False)
        }

        if len(successful_models) < 2:
            logger.warning("Not enough successful models for ensemble")
            return {}

        ensemble_results = {}

        # Voting Classifier
        try:
            voting_clf = VotingClassifier(
                estimators=list(successful_models.items()), voting="soft"
            )
            voting_clf.fit(data["X_train"], data["y_train"])

            metrics = self.evaluate_model(
                voting_clf, data["X_test"], data["y_test"], "voting_ensemble"
            )

            if metrics.get("meets_threshold", False):
                model_path = self.save_model(voting_clf, "voting_ensemble", metrics)
                metrics["model_path"] = str(model_path)

            ensemble_results["voting_ensemble"] = {
                "model": voting_clf,
                "metrics": metrics,
            }

            logger.info("✅ Voting ensemble created")

        except Exception as e:
            logger.error(f"❌ Failed to create voting ensemble: {e}")
            ensemble_results["voting_ensemble"] = {"error": str(e)}

        return ensemble_results

    def generate_training_report(
        self,
        baseline_results: Dict[str, Any],
        ensemble_results: Dict[str, Any],
        dataset_name: str = "healthcare_dataset",
    ):
        """Generate comprehensive training report."""
        logger.info("📊 Generating training report")

        all_results = {**baseline_results, **ensemble_results}

        # Performance summary
        performance_summary = []
        for model_name, result in all_results.items():
            if "metrics" in result:
                metrics = result["metrics"]
                performance_summary.append(
                    {
                        "model": model_name,
                        "roc_auc": metrics.get("roc_auc", 0),
                        "accuracy": metrics.get("accuracy", 0),
                        "f1_score": metrics.get("f1_score", 0),
                        "meets_threshold": metrics.get("meets_threshold", False),
                    }
                )

        # Find best model
        if performance_summary:
            best_model = max(performance_summary, key=lambda x: x["roc_auc"])
        else:
            best_model = None

        # Create report
        report = {
            "dataset": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_models_trained": len(all_results),
                "successful_models": len(
                    [
                        r
                        for r in all_results.values()
                        if "metrics" in r and r["metrics"].get("meets_threshold")
                    ]
                ),
                "best_model": best_model,
            },
            "baseline_models": baseline_results,
            "ensemble_models": ensemble_results,
            "performance_summary": performance_summary,
        }

        # Save report
        report_path = self.output_dir / f"{dataset_name}_training_report.json"
        import json

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Training report saved: {report_path}")

        # Print summary
        print("\n🏆 TRAINING SUMMARY")
        print("=" * 50)
        if best_model:
            print(f"Best Model: {best_model['model']}")
            print(f"ROC-AUC: {best_model['roc_auc']:.3f}")
            print(f"Accuracy: {best_model['accuracy']:.3f}")
            print(f"F1-Score: {best_model['f1_score']:.3f}")
        else:
            print("No successful models trained")

        print(f"\nModels meeting thresholds: {report['summary']['successful_models']}")
        print(f"Total models trained: {report['summary']['total_models_trained']}")

        return report


def create_sample_healthcare_data():
    """Create sample healthcare data for demonstration."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=2000,
        n_features=15,
        n_informative=10,
        n_redundant=3,
        n_classes=2,
        class_sep=0.8,
        random_state=42,
    )

    # Create meaningful feature names
    feature_names = [
        "age",
        "bmi",
        "blood_pressure_systolic",
        "blood_pressure_diastolic",
        "cholesterol_total",
        "cholesterol_hdl",
        "glucose_fasting",
        "heart_rate",
        "smoking_history",
        "exercise_frequency",
        "family_history",
        "previous_conditions",
        "medication_count",
        "stress_level",
        "sleep_quality",
    ]

    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="cardiovascular_risk")

    return X_df, y_series


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Enhanced Training Pipeline")
    parser.add_argument(
        "--config",
        default="config/operational_model_config.yaml",
        help="Configuration file path",
    )
    parser.add_argument(
        "--dataset", default="sample_healthcare", help="Dataset name for reporting"
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = EnhancedTrainingPipeline(args.config)

    logger.info("🏥 Enhanced Training Pipeline Starting")
    logger.info("=" * 60)

    try:
        # Create sample data (replace with real data loading)
        logger.info("📊 Loading healthcare data")
        X, y = create_sample_healthcare_data()
        logger.info(
            f"Dataset shape: {X.shape}, Target distribution: {y.value_counts().to_dict()}"
        )

        # Prepare data
        data = pipeline.prepare_data(X, y)
        logger.info("✅ Data preparation complete")

        # Train baseline models
        baseline_results = pipeline.train_baseline_models(data)

        # Create ensemble models
        ensemble_results = pipeline.create_ensemble_models(baseline_results, data)

        # Generate comprehensive report
        pipeline.generate_training_report(
            baseline_results, ensemble_results, args.dataset
        )

        logger.info("🎉 Enhanced training pipeline complete!")

    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
