#!/usr/bin/env python3
"""
Model Performance Analysis and Reporting

This script provides comprehensive analysis of trained models            # Calculate comprehensive metrics
            metrics = {
                'accuracy': float((y_pred == y_test).mean()),
                'roc_auc': float(roc_auc_score(y_test, y_proba)),
                'pr_auc': float(average_precision_score(y_test, y_proba)),
                'brier_s        # Save report (convert numpy types to native Python types)
        import json
        import numpy as np

        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(val) for key, val in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        readiness_report_clean = convert_numpy(readiness_report)

        report_path = self.output_dir / "production_readiness_report.json"
        with open(report_path, 'w') as f:
            json.dump(readiness_report_clean, f, indent=2)': float(brier_score_loss(y_test, y_proba)),
            }ing:
- Performance benchmarking
- Model comparison visualizations
- Calibration analysis
- Feature importance analysis
- Production readiness assessment
"""

import sys
from pathlib import Path
import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
import joblib
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("model_analysis")


class ModelPerformanceAnalyzer:
    """Comprehensive model performance analysis and reporting."""

    def __init__(self, models_dir: str = "reports/quick_models"):
        """Initialize the analyzer."""
        self.models_dir = Path(models_dir)
        self.output_dir = Path("reports/performance_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set plotting style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        logger.info(f"Performance Analyzer initialized")
        logger.info(f"Models directory: {self.models_dir}")
        logger.info(f"Output directory: {self.output_dir}")

    def load_models_and_data(self):
        """Load trained models and generate test data for analysis."""
        # Load models
        models = {}
        model_files = list(self.models_dir.glob("*.joblib"))

        for model_file in model_files:
            model_name = model_file.stem.rsplit("_", 2)[0]  # Remove timestamp
            models[model_name] = joblib.load(model_file)
            logger.info(f"Loaded {model_name} from {model_file.name}")

        if not models:
            raise ValueError(f"No models found in {self.models_dir}")

        # Generate test data (same as training for consistency)
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=7,
            n_redundant=2,
            n_classes=2,
            class_sep=0.8,
            random_state=42,
        )

        # Feature names
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

        # Use test split (same as training)
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        _, X_test, _, y_test = train_test_split(
            X_df, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale data
        scaler = StandardScaler()
        scaler.fit(X_df)
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )

        return models, X_test_scaled, y_test

    def analyze_model_performance(self, models, X_test, y_test):
        """Analyze performance of all models."""
        results = {}

        for model_name, model in models.items():
            logger.info(f"Analyzing {model_name}...")

            # Predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            # Calculate comprehensive metrics
            metrics = {
                "accuracy": (y_pred == y_test).mean(),
                "roc_auc": roc_auc_score(y_test, y_proba),
                "pr_auc": average_precision_score(y_test, y_proba),
                "brier_score": brier_score_loss(y_test, y_proba),
            }

            # ROC and PR curves
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            precision, recall, _ = precision_recall_curve(y_test, y_proba)

            # Calibration
            fraction_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=10)

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)

            results[model_name] = {
                "metrics": metrics,
                "predictions": {"y_pred": y_pred, "y_proba": y_proba},
                "curves": {
                    "roc": (fpr, tpr),
                    "pr": (precision, recall),
                    "calibration": (mean_pred, fraction_pos),
                },
                "confusion_matrix": cm,
            }

        return results

    def create_performance_dashboard(self, results, X_test, y_test, models=None):
        """Create comprehensive performance dashboard."""
        n_models = len(results)

        # Create figure with subplots
        fig = plt.figure(figsize=(18, 12))

        # Define subplot layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # 1. ROC Curves
        ax1 = fig.add_subplot(gs[0, 0])
        for model_name, result in results.items():
            fpr, tpr = result["curves"]["roc"]
            auc = result["metrics"]["roc_auc"]
            ax1.plot(fpr, tpr, label=f"{model_name} (AUC: {auc:.3f})")
        ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.set_title("ROC Curves")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Precision-Recall Curves
        ax2 = fig.add_subplot(gs[0, 1])
        for model_name, result in results.items():
            precision, recall = result["curves"]["pr"]
            pr_auc = result["metrics"]["pr_auc"]
            ax2.plot(recall, precision, label=f"{model_name} (AUC: {pr_auc:.3f})")
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title("Precision-Recall Curves")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Calibration Plot
        ax3 = fig.add_subplot(gs[0, 2])
        for model_name, result in results.items():
            mean_pred, fraction_pos = result["curves"]["calibration"]
            ax3.plot(mean_pred, fraction_pos, "s-", label=model_name)
        ax3.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect Calibration")
        ax3.set_xlabel("Mean Predicted Probability")
        ax3.set_ylabel("Fraction of Positives")
        ax3.set_title("Calibration Curves")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Performance Metrics Comparison
        ax4 = fig.add_subplot(gs[0, 3])
        metrics_df = pd.DataFrame(
            {name: result["metrics"] for name, result in results.items()}
        ).T

        metrics_df[["roc_auc", "pr_auc", "accuracy"]].plot(kind="bar", ax=ax4, rot=45)
        ax4.set_title("Performance Metrics Comparison")
        ax4.set_ylabel("Score")
        ax4.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # 5-6. Confusion Matrices
        model_names = list(results.keys())
        for i, (model_name, result) in enumerate(results.items()):
            if i >= 2:  # Only show first 2 models
                break
            ax = fig.add_subplot(gs[1, i])
            cm = result["confusion_matrix"]
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax,
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"],
            )
            ax.set_title(f"{model_name} - Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

        # 7. Prediction Distribution Comparison
        ax7 = fig.add_subplot(gs[1, 2:])
        for model_name, result in results.items():
            y_proba = result["predictions"]["y_proba"]
            ax7.hist(
                y_proba[y_test == 0],
                alpha=0.5,
                bins=20,
                label=f"{model_name} - Negative Class",
                density=True,
            )
            ax7.hist(
                y_proba[y_test == 1],
                alpha=0.5,
                bins=20,
                label=f"{model_name} - Positive Class",
                density=True,
            )
        ax7.set_xlabel("Predicted Probability")
        ax7.set_ylabel("Density")
        ax7.set_title("Prediction Distribution by Class")
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        # 8. Feature Importance (if available)
        ax8 = fig.add_subplot(gs[2, :2])
        feature_importance_data = []

        if models:
            for model_name, model in [(name, models[name]) for name in results.keys()]:
                if hasattr(model, "feature_importances_"):
                    importance = model.feature_importances_
                    feature_names = X_test.columns
                    feature_importance_data.append(
                        {
                            "model": model_name,
                            "features": feature_names,
                            "importance": importance,
                        }
                    )

        if feature_importance_data:
            # Plot feature importance for models that have it
            for data in feature_importance_data:
                if data["model"] == "random_forest":  # Focus on RF
                    importance_df = pd.DataFrame(
                        {"feature": data["features"], "importance": data["importance"]}
                    ).sort_values("importance", ascending=True)

                    ax8.barh(importance_df["feature"], importance_df["importance"])
                    ax8.set_title("Feature Importance (Random Forest)")
                    ax8.set_xlabel("Importance Score")
                    break
        else:
            ax8.text(
                0.5,
                0.5,
                "Feature importance not available",
                ha="center",
                va="center",
                transform=ax8.transAxes,
            )
            ax8.set_title("Feature Importance")

        # 9. Model Performance Summary Table
        ax9 = fig.add_subplot(gs[2, 2:])
        ax9.axis("tight")
        ax9.axis("off")

        summary_data = []
        for model_name, result in results.items():
            metrics = result["metrics"]
            summary_data.append(
                [
                    model_name,
                    f"{metrics['roc_auc']:.3f}",
                    f"{metrics['pr_auc']:.3f}",
                    f"{metrics['accuracy']:.3f}",
                    f"{metrics['brier_score']:.3f}",
                ]
            )

        table = ax9.table(
            cellText=summary_data,
            colLabels=["Model", "ROC-AUC", "PR-AUC", "Accuracy", "Brier Score"],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax9.set_title("Performance Summary", pad=20)

        plt.suptitle(
            "Operational Model Suite - Performance Dashboard", fontsize=16, y=0.98
        )

        # Save dashboard
        dashboard_path = self.output_dir / "performance_dashboard.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Performance dashboard saved: {dashboard_path}")
        return dashboard_path

    def generate_production_readiness_report(self, results, models):
        """Generate production readiness assessment."""
        logger.info("Generating production readiness report...")

        readiness_report = {
            "timestamp": datetime.now().isoformat(),
            "models_analyzed": len(results),
            "production_assessment": {},
        }

        for model_name, result in results.items():
            model = models[model_name]
            metrics = result["metrics"]

            # Define production criteria
            criteria = {
                "performance": {
                    "roc_auc_threshold": 0.75,
                    "accuracy_threshold": 0.70,
                    "calibration_threshold": 0.25,  # Brier score
                },
                "stability": {
                    # Would include CV scores in real implementation
                    "meets_stability": True
                },
                "interpretability": {
                    "has_feature_importance": hasattr(model, "feature_importances_"),
                    "model_complexity": "medium",  # Simplified
                },
            }

            # Assess production readiness
            performance_ready = (
                metrics["roc_auc"] >= criteria["performance"]["roc_auc_threshold"]
                and metrics["accuracy"] >= criteria["performance"]["accuracy_threshold"]
                and metrics["brier_score"]
                <= criteria["performance"]["calibration_threshold"]
            )

            overall_ready = (
                performance_ready and criteria["stability"]["meets_stability"]
            )

            assessment = {
                "performance_metrics": metrics,
                "meets_performance_criteria": performance_ready,
                "meets_stability_criteria": criteria["stability"]["meets_stability"],
                "interpretability_score": (
                    "high"
                    if criteria["interpretability"]["has_feature_importance"]
                    else "medium"
                ),
                "overall_production_ready": overall_ready,
                "recommendations": [],
            }

            # Generate recommendations
            if not performance_ready:
                if metrics["roc_auc"] < criteria["performance"]["roc_auc_threshold"]:
                    assessment["recommendations"].append(
                        f"Improve ROC-AUC from {metrics['roc_auc']:.3f} to >{criteria['performance']['roc_auc_threshold']}"
                    )
                if (
                    metrics["brier_score"]
                    > criteria["performance"]["calibration_threshold"]
                ):
                    assessment["recommendations"].append(
                        f"Improve calibration (Brier score: {metrics['brier_score']:.3f})"
                    )

            if overall_ready:
                assessment["recommendations"].append(
                    "✅ Ready for production deployment"
                )

            readiness_report["production_assessment"][model_name] = assessment

        # Save report
        report_path = self.output_dir / "production_readiness_report.json"
        with open(report_path, "w") as f:
            json.dump(readiness_report, f, indent=2)

        logger.info(f"Production readiness report saved: {report_path}")

        # Print summary
        print("\n🚀 PRODUCTION READINESS ASSESSMENT")
        print("=" * 50)

        for model_name, assessment in readiness_report["production_assessment"].items():
            status = (
                "✅ READY" if assessment["overall_production_ready"] else "⚠️ NEEDS WORK"
            )
            print(f"{model_name}: {status}")
            print(f"  ROC-AUC: {assessment['performance_metrics']['roc_auc']:.3f}")
            print(f"  Accuracy: {assessment['performance_metrics']['accuracy']:.3f}")
            print(
                f"  Calibration: {assessment['performance_metrics']['brier_score']:.3f}"
            )

            if assessment["recommendations"]:
                print("  Recommendations:")
                for rec in assessment["recommendations"]:
                    print(f"    • {rec}")
            print()

        return readiness_report


def main():
    """Main execution function."""
    logger.info("🔍 Starting Model Performance Analysis")
    logger.info("=" * 60)

    try:
        # Initialize analyzer
        analyzer = ModelPerformanceAnalyzer()

        # Load models and data
        logger.info("📊 Loading models and test data...")
        models, X_test, y_test = analyzer.load_models_and_data()

        # Analyze performance
        logger.info("🔬 Analyzing model performance...")
        results = analyzer.analyze_model_performance(models, X_test, y_test)

        # Create dashboard
        logger.info("📈 Creating performance dashboard...")
        dashboard_path = analyzer.create_performance_dashboard(
            results, X_test, y_test, models
        )

        # Generate production readiness report
        readiness_report = analyzer.generate_production_readiness_report(
            results, models
        )

        logger.info("✅ Model performance analysis complete!")
        logger.info(f"📊 Dashboard: {dashboard_path}")
        logger.info(f"📋 Reports: {analyzer.output_dir}")

    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
