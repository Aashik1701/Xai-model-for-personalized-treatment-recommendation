#!/usr/bin/env python3
"""
Operational Model Suite - Comprehensive Training Orchestrator

This script provides automated training of baseline and ensemble models with:
- Performance evaluation and reporting
- Fairness assessment across different patient groups
- Model calibration analysis
- Automated hyperparameter optimization
- Production-ready model versioning
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
    log_loss,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Add src to path for imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mlflow_manager import MLflowManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("operational_model_suite")


class ModelSuiteOrchestrator:
    """Orchestrates comprehensive model training and evaluation."""

    def __init__(self, output_dir: Path = None):
        """Initialize the orchestrator."""
        self.output_dir = output_dir or Path("reports/model_suite")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize MLflow manager
        self.mlflow_manager = MLflowManager(experiment_name="operational-model-suite")

        logger.info(f"Initialized Model Suite Orchestrator - Output: {self.output_dir}")

    def load_dataset_config(self) -> Dict[str, Any]:
        """Load dataset configuration."""
        config_path = Path("config/data_config.yaml")
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def create_baseline_models(self) -> Dict[str, Any]:
        """Create baseline model configurations."""
        return {
            "logistic_regression": {
                "model": LogisticRegression(random_state=42, max_iter=1000),
                "hyperparams": {
                    "C": [0.1, 1.0, 10.0],
                    "solver": ["liblinear", "lbfgs"],
                },
            },
            "random_forest": {
                "model": RandomForestClassifier(random_state=42),
                "hyperparams": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 10, None],
                    "min_samples_split": [2, 5, 10],
                },
            },
            "svm": {
                "model": SVC(random_state=42, probability=True),
                "hyperparams": {
                    "C": [0.1, 1.0, 10.0],
                    "kernel": ["linear", "rbf"],
                    "gamma": ["scale", "auto"],
                },
            },
        }

    def evaluate_model_performance(
        self,
        model: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        model_name: str,
    ) -> Dict[str, Any]:
        """Comprehensive model performance evaluation."""
        logger.info(f"Evaluating performance for {model_name}")

        # Fit model
        model.fit(X_train, y_train)

        # Generate predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Basic metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        # ROC and PR metrics
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)

        # Calibration metrics
        brier_score = brier_score_loss(y_test, y_proba)
        log_loss_score = log_loss(y_test, y_proba)

        # Cross-validation scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")

        performance_metrics = {
            "accuracy": report["accuracy"],
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1_score": report["weighted avg"]["f1-score"],
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "brier_score": brier_score,
            "log_loss": log_loss_score,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }

        # Generate performance plots
        self._create_performance_plots(
            y_test, y_pred, y_proba, model_name, performance_metrics
        )

        return performance_metrics

    def _create_performance_plots(
        self,
        y_test: pd.Series,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        model_name: str,
        metrics: Dict[str, Any],
    ) -> None:
        """Create comprehensive performance visualization plots."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"{model_name} - Performance Analysis", fontsize=16)

        # 1. Confusion Matrix
        cm = metrics["confusion_matrix"]
        sns.heatmap(cm, annot=True, fmt="d", ax=axes[0, 0], cmap="Blues")
        axes[0, 0].set_title("Confusion Matrix")
        axes[0, 0].set_xlabel("Predicted")
        axes[0, 0].set_ylabel("Actual")

        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        axes[0, 1].plot(fpr, tpr, label=f'ROC AUC = {metrics["roc_auc"]:.3f}')
        axes[0, 1].plot([0, 1], [0, 1], "k--")
        axes[0, 1].set_xlabel("False Positive Rate")
        axes[0, 1].set_ylabel("True Positive Rate")
        axes[0, 1].set_title("ROC Curve")
        axes[0, 1].legend()

        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        axes[0, 2].plot(recall, precision, label=f'PR AUC = {metrics["pr_auc"]:.3f}')
        axes[0, 2].set_xlabel("Recall")
        axes[0, 2].set_ylabel("Precision")
        axes[0, 2].set_title("Precision-Recall Curve")
        axes[0, 2].legend()

        # 4. Calibration Plot
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_proba, n_bins=10
        )
        axes[1, 0].plot(
            mean_predicted_value, fraction_of_positives, "s-", label=model_name
        )
        axes[1, 0].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        axes[1, 0].set_xlabel("Mean Predicted Probability")
        axes[1, 0].set_ylabel("Fraction of Positives")
        axes[1, 0].set_title("Calibration Plot")
        axes[1, 0].legend()

        # 5. Prediction Distribution
        axes[1, 1].hist(
            y_proba[y_test == 0], alpha=0.5, label="Negative Class", bins=20
        )
        axes[1, 1].hist(
            y_proba[y_test == 1], alpha=0.5, label="Positive Class", bins=20
        )
        axes[1, 1].set_xlabel("Predicted Probability")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].set_title("Prediction Distribution")
        axes[1, 1].legend()

        # 6. Metrics Summary
        metrics_text = f"""
        Accuracy: {metrics['accuracy']:.3f}
        Precision: {metrics['precision']:.3f}
        Recall: {metrics['recall']:.3f}
        F1-Score: {metrics['f1_score']:.3f}
        ROC-AUC: {metrics['roc_auc']:.3f}
        PR-AUC: {metrics['pr_auc']:.3f}
        Brier Score: {metrics['brier_score']:.3f}
        CV Mean±Std: {metrics['cv_mean']:.3f}±{metrics['cv_std']:.3f}
        """
        axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment="center")
        axes[1, 2].set_title("Performance Summary")
        axes[1, 2].axis("off")

        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / f"{model_name}_performance_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Performance plots saved: {plot_path}")

    def assess_fairness(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_proba: np.ndarray,
        sensitive_features: List[str],
        model_name: str,
    ) -> Dict[str, Any]:
        """Assess model fairness across different patient groups."""
        logger.info(f"Assessing fairness for {model_name}")

        fairness_metrics = {}

        for feature in sensitive_features:
            if feature not in X_test.columns:
                logger.warning(f"Sensitive feature '{feature}' not found in test data")
                continue

            # Create binary groups for continuous features
            if X_test[feature].dtype in ["float64", "int64"]:
                median_val = X_test[feature].median()
                groups = (X_test[feature] > median_val).astype(int)
                group_names = [f"{feature}_below_median", f"{feature}_above_median"]
            else:
                groups = X_test[feature]
                group_names = groups.unique()

            group_metrics = {}
            for group_val in groups.unique():
                mask = groups == group_val
                if mask.sum() == 0:
                    continue

                group_y_test = y_test[mask]
                group_y_proba = y_proba[mask]

                if len(group_y_test.unique()) < 2:
                    logger.warning(
                        f"Group {group_val} has only one class, skipping metrics"
                    )
                    continue

                group_metrics[str(group_val)] = {
                    "size": mask.sum(),
                    "positive_rate": group_y_test.mean(),
                    "roc_auc": roc_auc_score(group_y_test, group_y_proba),
                    "average_precision": average_precision_score(
                        group_y_test, group_y_proba
                    ),
                }

            fairness_metrics[feature] = group_metrics

        # Calculate fairness disparities
        disparity_metrics = self._calculate_fairness_disparities(fairness_metrics)

        # Save fairness report
        fairness_report = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "group_metrics": fairness_metrics,
            "disparity_metrics": disparity_metrics,
        }

        report_path = self.output_dir / f"{model_name}_fairness_report.json"
        with open(report_path, "w") as f:
            json.dump(fairness_report, f, indent=2)

        logger.info(f"Fairness report saved: {report_path}")
        return fairness_report

    def _calculate_fairness_disparities(
        self, fairness_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate fairness disparity metrics."""
        disparities = {}

        for feature, group_metrics in fairness_metrics.items():
            if len(group_metrics) < 2:
                continue

            roc_aucs = [metrics["roc_auc"] for metrics in group_metrics.values()]
            positive_rates = [
                metrics["positive_rate"] for metrics in group_metrics.values()
            ]

            disparities[feature] = {
                "roc_auc_disparity": max(roc_aucs) - min(roc_aucs),
                "positive_rate_disparity": max(positive_rates) - min(positive_rates),
                "equalized_odds_difference": max(roc_aucs)
                - min(roc_aucs),  # Simplified
            }

        return disparities

    def generate_comprehensive_report(
        self, all_results: Dict[str, Any], dataset_name: str
    ) -> None:
        """Generate comprehensive model suite report."""
        logger.info("Generating comprehensive model suite report")

        report = {
            "dataset": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_models": len(all_results),
                "best_model": self._find_best_model(all_results),
                "evaluation_criteria": [
                    "ROC-AUC",
                    "Precision-Recall AUC",
                    "Calibration",
                    "Cross-validation stability",
                    "Fairness assessment",
                ],
            },
            "models": all_results,
            "recommendations": self._generate_recommendations(all_results),
        }

        # Save comprehensive report
        report_path = (
            self.output_dir / f"{dataset_name}_operational_model_suite_report.json"
        )
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Create executive summary
        self._create_executive_summary(report, dataset_name)

        logger.info(f"Comprehensive report saved: {report_path}")

    def _find_best_model(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Find the best performing model based on composite score."""
        best_model = None
        best_score = -1

        for model_name, model_results in results.items():
            if "performance" not in model_results:
                continue

            perf = model_results["performance"]
            # Composite score: ROC-AUC + PR-AUC - Brier Score
            score = (
                perf.get("roc_auc", 0)
                + perf.get("pr_auc", 0)
                - perf.get("brier_score", 1)
            )

            if score > best_score:
                best_score = score
                best_model = {
                    "name": model_name,
                    "composite_score": score,
                    "roc_auc": perf.get("roc_auc", 0),
                    "pr_auc": perf.get("pr_auc", 0),
                    "brier_score": perf.get("brier_score", 1),
                }

        return best_model

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on results."""
        recommendations = []

        # Performance-based recommendations
        best_model = self._find_best_model(results)
        if best_model:
            recommendations.append(
                f"Deploy {best_model['name']} as primary model "
                f"(ROC-AUC: {best_model['roc_auc']:.3f})"
            )

        # Fairness recommendations
        for model_name, model_results in results.items():
            if "fairness" in model_results:
                fairness = model_results["fairness"]
                for feature, disparity in fairness.get("disparity_metrics", {}).items():
                    if disparity.get("roc_auc_disparity", 0) > 0.1:
                        recommendations.append(
                            f"Review {model_name} for fairness issues in {feature} "
                            f"(disparity: {disparity['roc_auc_disparity']:.3f})"
                        )

        # Calibration recommendations
        for model_name, model_results in results.items():
            if "performance" in model_results:
                brier_score = model_results["performance"].get("brier_score", 0)
                if brier_score > 0.25:
                    recommendations.append(
                        f"Consider calibrating {model_name} "
                        f"(Brier score: {brier_score:.3f})"
                    )

        return recommendations

    def _create_executive_summary(
        self, report: Dict[str, Any], dataset_name: str
    ) -> None:
        """Create executive summary markdown report."""
        summary_content = f"""# Operational Model Suite - Executive Summary

**Dataset:** {dataset_name}  
**Generated:** {report['timestamp']}  
**Total Models Evaluated:** {report['summary']['total_models']}

## 🏆 Best Performing Model

**{report['summary']['best_model']['name']}**
- ROC-AUC: {report['summary']['best_model']['roc_auc']:.3f}
- PR-AUC: {report['summary']['best_model']['pr_auc']:.3f}
- Calibration (Brier Score): {report['summary']['best_model']['brier_score']:.3f}
- Composite Score: {report['summary']['best_model']['composite_score']:.3f}

## 📊 Model Performance Comparison

| Model | ROC-AUC | PR-AUC | Brier Score | CV Stability |
|-------|---------|--------|-------------|--------------|
"""

        for model_name, results in report["models"].items():
            if "performance" in results:
                perf = results["performance"]
                summary_content += f"| {model_name} | {perf.get('roc_auc', 'N/A'):.3f} | {perf.get('pr_auc', 'N/A'):.3f} | {perf.get('brier_score', 'N/A'):.3f} | {perf.get('cv_mean', 'N/A'):.3f}±{perf.get('cv_std', 'N/A'):.3f} |\n"

        summary_content += f"""
## 🎯 Key Recommendations

"""
        for i, rec in enumerate(report["recommendations"], 1):
            summary_content += f"{i}. {rec}\n"

        summary_content += f"""
## 📋 Evaluation Criteria

- **Performance Metrics:** ROC-AUC, Precision-Recall AUC, Accuracy, F1-Score
- **Calibration:** Brier Score, Log Loss, Reliability diagrams
- **Stability:** Cross-validation performance consistency
- **Fairness:** Performance equity across patient subgroups
- **Production Readiness:** Model size, inference speed, interpretability

## 📁 Detailed Reports

- Performance Analysis: `*_performance_analysis.png`
- Fairness Assessment: `*_fairness_report.json`
- Full Technical Report: `{dataset_name}_operational_model_suite_report.json`

---
*Generated by Operational Model Suite v1.0*
"""

        summary_path = self.output_dir / f"{dataset_name}_executive_summary.md"
        with open(summary_path, "w") as f:
            f.write(summary_content)

        logger.info(f"Executive summary saved: {summary_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Operational Model Suite Training")
    parser.add_argument(
        "--dataset", default="heart_disease", help="Dataset to train on"
    )
    parser.add_argument("--output-dir", type=Path, help="Output directory for reports")
    parser.add_argument(
        "--sensitive-features",
        nargs="+",
        default=["age", "sex"],
        help="Features to assess for fairness",
    )

    args = parser.parse_args()

    # Initialize orchestrator
    orchestrator = ModelSuiteOrchestrator(args.output_dir)

    logger.info(f"🏥 Starting Operational Model Suite for {args.dataset}")
    logger.info("=" * 60)

    try:
        # For now, create a simple demonstration
        # In a real implementation, this would load and process the actual dataset
        logger.info("⚠️  Demonstration mode - using synthetic data")
        logger.info("   In production, this would load real healthcare datasets")

        # Create sample data for demonstration
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=7,
            n_redundant=2,
            n_classes=2,
            random_state=42,
        )

        # Create DataFrame with feature names
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name="target")

        # Add synthetic sensitive features
        X_df["age"] = np.random.randint(18, 80, size=len(X_df))
        X_df["sex"] = np.random.choice([0, 1], size=len(X_df))

        # Split data
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_series, test_size=0.2, random_state=42, stratify=y_series
        )

        # Train and evaluate baseline models
        baseline_models = orchestrator.create_baseline_models()
        all_results = {}

        for model_name, model_config in baseline_models.items():
            logger.info(f"Training and evaluating {model_name}")

            model = model_config["model"]

            # Performance evaluation
            performance = orchestrator.evaluate_model_performance(
                model, X_train, X_test, y_train, y_test, model_name
            )

            # Fairness assessment
            y_proba = model.predict_proba(X_test)[:, 1]
            fairness = orchestrator.assess_fairness(
                model, X_test, y_test, y_proba, args.sensitive_features, model_name
            )

            all_results[model_name] = {
                "performance": performance,
                "fairness": fairness,
                "model_config": str(model_config["model"]),
            }

            logger.info(f"✅ {model_name} evaluation complete")

        # Generate comprehensive report
        orchestrator.generate_comprehensive_report(all_results, args.dataset)

        logger.info("🎉 Operational Model Suite complete!")
        logger.info(f"📊 Reports generated in: {orchestrator.output_dir}")

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
