"""
Full Training Pipeline with MLflow Tracking

Train models on all 13 real healthcare datasets with comprehensive MLflow experiment tracking.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import logging
import mlflow
import mlflow.sklearn
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MLflowTrainingPipeline:
    """Comprehensive training pipeline with MLflow tracking."""

    def __init__(self, mlflow_uri: str = "http://127.0.0.1:5000"):
        """Initialize pipeline with MLflow tracking."""
        self.mlflow_uri = mlflow_uri
        mlflow.set_tracking_uri(mlflow_uri)

        # Create experiment
        experiment_name = (
            f"Real_Datasets_Training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except:
            self.experiment_id = mlflow.get_experiment_by_name(
                experiment_name
            ).experiment_id

        mlflow.set_experiment(experiment_name)
        logger.info(f"📊 MLflow Experiment: {experiment_name}")
        logger.info(f"🔗 Tracking URI: {mlflow_uri}")

        # Dataset configurations
        self.datasets = [
            "heart_disease_uci",
            "heart_disease_multicenter",
            "breast_cancer_wisconsin",
            "diabetes_130_hospitals",
            "hepatitis",
            "dermatology",
            "medical_appointments",
            "covid_symptoms",
            "thyroid_disease",
            "synthea_longitudinal",
            "skin_cancer_imaging",
            "medical_question_pairs",
            "drug_reviews",
        ]

        # Model configurations
        self.models = {
            "RandomForest": RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=100, max_depth=5, random_state=42
            ),
            "LogisticRegression": LogisticRegression(
                random_state=42, max_iter=1000, n_jobs=-1
            ),
        }

        self.results = []

    def load_dataset(
        self, dataset_name: str
    ) -> Optional[Tuple[pd.DataFrame, pd.Series, List[str]]]:
        """Load and prepare dataset."""
        data_path = Path(f"data/processed/{dataset_name}_processed.csv")

        if not data_path.exists():
            logger.warning(f"⚠️  Dataset not found: {data_path}")
            return None

        try:
            df = pd.read_csv(data_path)
            logger.info(f"✅ Loaded {dataset_name}: {df.shape}")

            # Determine target column
            target_col = None
            for possible_target in ["target", "is_duplicate", "rating"]:
                if possible_target in df.columns:
                    target_col = possible_target
                    break

            if target_col is None:
                logger.error(f"❌ No target column found in {dataset_name}")
                return None

            # Prepare features
            exclude_cols = [
                target_col,
                "patient_id",
                "image_path",
                "lesion_type",
                "icd10_code",
                "lesion_category",
                "condition_summary",
                "medication_summary",
                "dominant_condition_domain",
                "temporal_event_summary",
                "birthdate",
                "deathdate",
                "GENDER",
                "RACE",
                "ETHNICITY",
                "ZIP",
                "CITY",
                "STATE",
                "MARITAL",
                "split",
                "question_primary",
                "question_secondary",
                "drugName",
                "condition",
                "review",
            ]

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numeric_cols if col not in exclude_cols]

            X = df[feature_cols].fillna(0)
            y = df[target_col]

            return X, y, feature_cols

        except Exception as e:
            logger.error(f"❌ Error loading {dataset_name}: {e}")
            return None

    def train_model(
        self,
        dataset_name: str,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: List[str],
    ) -> Dict:
        """Train a single model with MLflow tracking."""

        with mlflow.start_run(run_name=f"{dataset_name}_{model_name}"):
            # Log dataset info
            mlflow.log_param("dataset", dataset_name)
            mlflow.log_param("model", model_name)
            mlflow.log_param("n_samples", len(X))
            mlflow.log_param("n_features", len(feature_cols))
            mlflow.log_param("target_distribution", str(y.value_counts().to_dict()))

            # Split data
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            except ValueError:
                # If stratification fails (too few samples in a class)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model
            model = self.models[model_name]
            logger.info(f"  🔧 Training {model_name}...")

            start_time = datetime.now()
            model.fit(X_train_scaled, y_train)
            training_time = (datetime.now() - start_time).total_seconds()

            mlflow.log_metric("training_time_seconds", training_time)

            # Evaluate on test set
            y_pred = model.predict(X_test_scaled)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            )
            recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            # ROC AUC for binary classification
            if len(y.unique()) == 2:
                try:
                    if hasattr(model, "predict_proba"):
                        y_proba = model.predict_proba(X_test_scaled)[:, 1]
                        roc_auc = roc_auc_score(y_test, y_proba)
                        mlflow.log_metric("roc_auc", roc_auc)
                except:
                    pass

            # Cross-validation score
            try:
                cv_scores = cross_val_score(
                    model, X_train_scaled, y_train, cv=5, n_jobs=-1
                )
                mlflow.log_metric("cv_mean", cv_scores.mean())
                mlflow.log_metric("cv_std", cv_scores.std())
            except:
                pass

            # Feature importance (for tree-based models)
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                top_features = sorted(
                    zip(feature_cols, importances), key=lambda x: x[1], reverse=True
                )[:10]

                for rank, (feat, imp) in enumerate(top_features, 1):
                    mlflow.log_metric(f"feature_importance_rank_{rank}", imp)
                    mlflow.log_param(f"top_feature_{rank}", feat)

            # Save model
            model_dir = Path("models/trained_real_datasets")
            model_dir.mkdir(parents=True, exist_ok=True)

            model_path = model_dir / f"{dataset_name}_{model_name}.joblib"
            joblib.dump(
                {
                    "model": model,
                    "scaler": scaler,
                    "feature_names": feature_cols,
                    "accuracy": accuracy,
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                },
                model_path,
            )

            # Log model to MLflow
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_artifact(str(model_path))

            logger.info(f"  ✅ {model_name}: Accuracy={accuracy:.4f}, F1={f1:.4f}")

            return {
                "dataset": dataset_name,
                "model": model_name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "training_time": training_time,
                "n_features": len(feature_cols),
                "n_samples": len(X),
            }

    def train_all(self) -> List[Dict]:
        """Train all models on all datasets."""

        logger.info("🚀 Starting Full Training Pipeline on 13 Datasets")
        logger.info("=" * 80)

        for dataset_name in self.datasets:
            logger.info(f"\n📊 Processing: {dataset_name.upper()}")
            logger.info("-" * 80)

            # Load dataset
            data = self.load_dataset(dataset_name)
            if data is None:
                continue

            X, y, feature_cols = data

            # Train multiple models
            for model_name in self.models.keys():
                try:
                    result = self.train_model(
                        dataset_name, model_name, X, y, feature_cols
                    )
                    self.results.append(result)
                except Exception as e:
                    logger.error(
                        f"  ❌ Failed to train {model_name} on {dataset_name}: {e}"
                    )
                    continue

        return self.results

    def generate_summary_report(self) -> pd.DataFrame:
        """Generate summary report of all training results."""

        if not self.results:
            logger.warning("No results to summarize")
            return pd.DataFrame()

        df_results = pd.DataFrame(self.results)

        # Save results
        report_dir = Path("reports/training_results")
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"training_summary_{timestamp}.csv"
        df_results.to_csv(report_path, index=False)

        logger.info(f"\n💾 Training summary saved: {report_path}")

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("📈 TRAINING SUMMARY")
        logger.info("=" * 80)

        logger.info(f"\nTotal models trained: {len(df_results)}")
        logger.info(f"Datasets processed: {df_results['dataset'].nunique()}")
        logger.info(f"Model types: {df_results['model'].nunique()}")

        logger.info("\n🏆 Top 10 Models by Accuracy:")
        top_models = df_results.nlargest(10, "accuracy")[
            ["dataset", "model", "accuracy", "f1_score"]
        ]
        print(top_models.to_string(index=False))

        logger.info("\n📊 Average Performance by Model Type:")
        avg_by_model = df_results.groupby("model")[
            ["accuracy", "precision", "recall", "f1_score"]
        ].mean()
        print(avg_by_model.to_string())

        logger.info("\n📊 Average Performance by Dataset:")
        avg_by_dataset = (
            df_results.groupby("dataset")[["accuracy", "f1_score"]]
            .mean()
            .sort_values("accuracy", ascending=False)
        )
        print(avg_by_dataset.to_string())

        return df_results


def main():
    """Main execution function."""

    print("\n" + "=" * 80)
    print("🏥 XAI HEALTHCARE MODEL TRAINING PIPELINE")
    print("=" * 80)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Datasets: 13 (including Synthea, ISIC, NLP datasets)")
    print(f"🤖 Models: 3 (RandomForest, GradientBoosting, LogisticRegression)")
    print(f"📈 Expected Experiments: ~39 MLflow runs")
    print("=" * 80 + "\n")

    # Initialize pipeline
    pipeline = MLflowTrainingPipeline(mlflow_uri="http://127.0.0.1:5000")

    # Train all models
    results = pipeline.train_all()

    # Generate summary
    summary_df = pipeline.generate_summary_report()

    # Final summary
    print("\n" + "=" * 80)
    print("✅ TRAINING PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"📊 Total experiments: {len(results)}")
    print(f"💾 Models saved to: models/trained_real_datasets/")
    print(f"📈 MLflow UI: http://127.0.0.1:5000")
    print(f"📁 Summary report: reports/training_results/")
    print("\n🚀 Next Steps:")
    print("  1. View MLflow dashboard: http://127.0.0.1:5000")
    print("  2. Compare model performance across datasets")
    print("  3. Test explainability: python scripts/simple_explainer.py")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
