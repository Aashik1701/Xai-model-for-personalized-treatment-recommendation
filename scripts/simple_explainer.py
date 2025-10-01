#!/usr/bin/env python3
"""
Simple Explainability Toolkit - Production Ready

A simplified but robust implementation of model explainability
for healthcare machine learning models.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import json
import warnings

warnings.filterwarnings("ignore")

# ML libraries
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from sklearn.inspection import permutation_importance

# Project imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from mlflow_manager import MLflowManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("simple_explainer")


class SimpleExplainer:
    """Simple but effective model explainer."""

    def __init__(self, output_dir: str = "reports/explanations"):
        """Initialize the explainer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.mlflow_manager = MLflowManager(experiment_name="simple-explanations")

        logger.info(f"Simple Explainer initialized - Output: {self.output_dir}")

    def load_model_and_data(self, model_path: str):
        """Load model and create test data."""
        logger.info(f"Loading model from {model_path}")

        model = joblib.load(model_path)

        # Generate consistent test data
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=7,
            n_redundant=2,
            n_classes=2,
            class_sep=0.8,
            random_state=42,
        )

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

        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )

        return model, X_test_scaled, y_test

    def calculate_feature_importance(self, model, X_test, y_test):
        """Calculate multiple types of feature importance."""
        results = {}

        # Built-in importance (for tree models)
        if hasattr(model, "feature_importances_"):
            results["builtin"] = {
                "importance": model.feature_importances_.tolist(),
                "feature_names": X_test.columns.tolist(),
            }
            logger.info("✅ Built-in feature importance calculated")

        # Permutation importance
        try:
            perm_importance = permutation_importance(
                model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
            )
            results["permutation"] = {
                "importance": perm_importance.importances_mean.tolist(),
                "std": perm_importance.importances_std.tolist(),
                "feature_names": X_test.columns.tolist(),
            }
            logger.info("✅ Permutation importance calculated")
        except Exception as e:
            logger.error(f"Permutation importance failed: {e}")

        # SHAP (if available)
        if SHAP_AVAILABLE:
            try:
                if hasattr(model, "estimators_"):  # Tree-based models
                    explainer = shap.TreeExplainer(model)
                    sample_size = min(50, len(X_test))
                    X_sample = X_test.sample(n=sample_size, random_state=42)

                    shap_values = explainer.shap_values(X_sample)

                    # Handle multi-class output
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]  # Positive class

                    # Ensure we have a 2D array (samples x features)
                    if len(shap_values.shape) > 2:
                        # For multi-output models, take the first output
                        if shap_values.shape[2] > 0:
                            shap_values = shap_values[:, :, 0]

                    # Calculate feature importance as mean absolute SHAP values
                    feature_importance = np.abs(shap_values).mean(axis=0)
                    # Ensure it's 1D
                    if len(feature_importance.shape) > 1:
                        feature_importance = feature_importance.flatten()

                    results["shap"] = {
                        "values": shap_values.tolist(),
                        "feature_importance": feature_importance.tolist(),
                        "feature_names": X_sample.columns.tolist(),
                        "expected_value": float(
                            explainer.expected_value[1]
                            if isinstance(explainer.expected_value, np.ndarray)
                            else explainer.expected_value
                        ),
                    }
                    logger.info("✅ SHAP explanations calculated")

            except Exception as e:
                logger.error(f"SHAP calculation failed: {e}")

        return results

    def create_visualizations(self, importance_results, model_name):
        """Create explanation visualizations."""
        saved_plots = []

        # Feature importance comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"{model_name} - Feature Importance Analysis", fontsize=16)

        # Ensure axes is always a 2D array for consistent indexing
        if len(axes.shape) == 1:
            axes = axes.reshape(2, 2)

        # 1. Built-in importance (if available)
        if "builtin" in importance_results:
            builtin = importance_results["builtin"]
            feature_names = builtin["feature_names"]
            importance = np.array(builtin["importance"])

            # Sort by importance
            sorted_idx = np.argsort(importance)

            axes[0, 0].barh(range(len(feature_names)), importance[sorted_idx])
            axes[0, 0].set_yticks(range(len(feature_names)))
            axes[0, 0].set_yticklabels([feature_names[i] for i in sorted_idx])
            axes[0, 0].set_xlabel("Feature Importance")
            axes[0, 0].set_title("Built-in Feature Importance")
            axes[0, 0].grid(True, alpha=0.3)

        # 2. Permutation importance
        if "permutation" in importance_results:
            perm = importance_results["permutation"]
            feature_names = perm["feature_names"]
            importance = np.array(perm["importance"])

            sorted_idx = np.argsort(importance)

            axes[0, 1].barh(range(len(feature_names)), importance[sorted_idx])
            axes[0, 1].set_yticks(range(len(feature_names)))
            axes[0, 1].set_yticklabels([feature_names[i] for i in sorted_idx])
            axes[0, 1].set_xlabel("Permutation Importance")
            axes[0, 1].set_title("Permutation Feature Importance")
            axes[0, 1].grid(True, alpha=0.3)

        # 3. SHAP importance (if available)
        if "shap" in importance_results:
            shap_data = importance_results["shap"]
            feature_names = shap_data["feature_names"]
            importance = np.array(shap_data["feature_importance"])

            sorted_idx = np.argsort(importance)

            axes[1, 0].barh(range(len(feature_names)), importance[sorted_idx])
            axes[1, 0].set_yticks(range(len(feature_names)))
            axes[1, 0].set_yticklabels([feature_names[i] for i in sorted_idx])
            axes[1, 0].set_xlabel("SHAP Importance")
            axes[1, 0].set_title("SHAP Feature Importance")
            axes[1, 0].grid(True, alpha=0.3)

        # 4. Comparison plot
        comparison_data = {}
        for method, data in importance_results.items():
            if method == "builtin":
                comparison_data["Built-in"] = data["importance"]
            elif method == "permutation":
                comparison_data["Permutation"] = data["importance"]
            elif method == "shap":
                comparison_data["SHAP"] = data["feature_importance"]

        if comparison_data:
            feature_names = importance_results[list(importance_results.keys())[0]][
                "feature_names"
            ]
            comparison_df = pd.DataFrame(comparison_data, index=feature_names)

            # Top 8 features for comparison
            if len(comparison_df) > 8:
                # Get top features from first method
                first_method = list(comparison_data.keys())[0]
                top_features = comparison_df[first_method].nlargest(8).index
                comparison_df = comparison_df.loc[top_features]

            comparison_df.plot(kind="bar", ax=axes[1, 1], rot=45)
            axes[1, 1].set_title("Feature Importance Comparison")
            axes[1, 1].set_xlabel("Features")
            axes[1, 1].set_ylabel("Importance Score")
            axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / f"{model_name}_feature_importance.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        saved_plots.append(str(plot_path))

        # Create clinical interpretation
        clinical_path = self._create_clinical_summary(importance_results, model_name)
        if clinical_path:
            saved_plots.append(clinical_path)

        return saved_plots

    def _create_clinical_summary(self, importance_results, model_name):
        """Create clinical interpretation summary."""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Get the best available importance method
        importance_data = None
        method_name = ""

        if "shap" in importance_results:
            importance_data = dict(
                zip(
                    importance_results["shap"]["feature_names"],
                    importance_results["shap"]["feature_importance"],
                )
            )
            method_name = "SHAP"
        elif "builtin" in importance_results:
            importance_data = dict(
                zip(
                    importance_results["builtin"]["feature_names"],
                    importance_results["builtin"]["importance"],
                )
            )
            method_name = "Built-in"
        elif "permutation" in importance_results:
            importance_data = dict(
                zip(
                    importance_results["permutation"]["feature_names"],
                    importance_results["permutation"]["importance"],
                )
            )
            method_name = "Permutation"

        if not importance_data:
            return None

        # Clinical groupings
        clinical_groups = {
            "Demographics": ["age"],
            "Physical Health": ["bmi", "blood_pressure", "heart_rate"],
            "Laboratory": ["glucose", "cholesterol"],
            "Lifestyle": ["smoking", "exercise", "stress_level"],
            "Medical History": ["family_history"],
        }

        # Group importance
        grouped_importance = {}
        for group_name, features in clinical_groups.items():
            group_total = sum(importance_data.get(feature, 0) for feature in features)
            if group_total > 0:
                grouped_importance[group_name] = group_total

        # Create pie chart
        if grouped_importance:
            plt.pie(
                grouped_importance.values(),
                labels=grouped_importance.keys(),
                autopct="%1.1f%%",
                startangle=90,
            )
            plt.title(
                f"{model_name} - Clinical Risk Factor Groups\n({method_name} Importance)",
                fontsize=14,
            )

            # Add clinical interpretation text
            top_group = max(grouped_importance.items(), key=lambda x: x[1])
            interpretation = f"Primary risk domain: {top_group[0]}\n"
            interpretation += f"Contributing {top_group[1]/sum(grouped_importance.values())*100:.1f}% "
            interpretation += "of model decision factors"

            plt.figtext(
                0.02,
                0.02,
                interpretation,
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
            )

            clinical_path = (
                self.output_dir / f"{model_name}_clinical_interpretation.png"
            )
            plt.savefig(clinical_path, dpi=300, bbox_inches="tight")
            plt.close()

            return str(clinical_path)

        plt.close()
        return None

    def generate_report(self, model_name, importance_results, visualizations):
        """Generate explanation report."""
        logger.info(f"Generating explanation report for {model_name}")

        # Find top features across methods
        all_features = {}
        for method, data in importance_results.items():
            features = data["feature_names"]
            if method == "builtin":
                importances = data["importance"]
            elif method == "permutation":
                importances = data["importance"]
            elif method == "shap":
                importances = data["feature_importance"]
            else:
                continue

            for feature, importance in zip(features, importances):
                if feature not in all_features:
                    all_features[feature] = {}
                all_features[feature][method] = float(importance)

        # Calculate average importance
        feature_avg_importance = {}
        for feature, methods in all_features.items():
            if methods:
                feature_avg_importance[feature] = sum(methods.values()) / len(methods)

        # Top 5 features
        top_features = sorted(
            feature_avg_importance.items(), key=lambda x: x[1], reverse=True
        )[:5]

        report = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "methods_used": list(importance_results.keys()),
            "top_features": [
                {"feature": feature, "avg_importance": importance}
                for feature, importance in top_features
            ],
            "detailed_results": importance_results,
            "visualizations": visualizations,
            "clinical_insights": self._generate_clinical_insights(top_features),
        }

        # Save report
        report_path = self.output_dir / f"{model_name}_explanation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved: {report_path}")
        return str(report_path)

    def _generate_clinical_insights(self, top_features):
        """Generate clinical insights from top features."""
        insights = []

        if not top_features:
            return insights

        # Feature categories
        categories = {
            "modifiable": ["smoking", "exercise", "stress_level", "bmi"],
            "medical": ["blood_pressure", "cholesterol", "glucose", "heart_rate"],
            "demographic": ["age"],
            "genetic": ["family_history"],
        }

        # Categorize top features
        top_feature_names = [f[0] for f in top_features[:3]]

        for category, features in categories.items():
            category_features = [f for f in top_feature_names if f in features]
            if category_features:
                if category == "modifiable":
                    insights.append(
                        f"Modifiable risk factors identified: {', '.join(category_features)}. "
                        "Patient education and lifestyle interventions may be effective."
                    )
                elif category == "medical":
                    insights.append(
                        f"Medical monitoring needed for: {', '.join(category_features)}. "
                        "Consider regular clinical assessment."
                    )
                elif category == "demographic":
                    insights.append(
                        "Age is a significant factor. Consider age-appropriate screening protocols."
                    )
                elif category == "genetic":
                    insights.append(
                        "Family history is influential. Genetic counseling may be beneficial."
                    )

        return insights

    def explain_model(self, model_path, model_name=None):
        """Generate comprehensive model explanations."""
        if model_name is None:
            model_name = Path(model_path).stem.split("_")[0]

        logger.info(f"🔍 Explaining model: {model_name}")

        # Load model and data
        model, X_test, y_test = self.load_model_and_data(model_path)

        # Calculate importance
        importance_results = self.calculate_feature_importance(model, X_test, y_test)

        # Create visualizations
        visualizations = self.create_visualizations(importance_results, model_name)

        # Generate report
        report_path = self.generate_report(
            model_name, importance_results, visualizations
        )

        # Log to MLflow
        with self.mlflow_manager.start_run(run_name=f"{model_name}_explanations"):
            metrics = {
                "methods_used": len(importance_results),
                "features_analyzed": len(X_test.columns),
                "visualizations_created": len(visualizations),
            }
            self.mlflow_manager.log_model_metrics(metrics, prefix="explanation")

        logger.info(f"✅ Explanation complete for {model_name}")

        return {
            "model_name": model_name,
            "importance_results": importance_results,
            "visualizations": visualizations,
            "report_path": report_path,
        }


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Simple Model Explainer")
    parser.add_argument("--model-path", required=True, help="Path to model file")
    parser.add_argument("--model-name", help="Model name")
    args = parser.parse_args()

    logger.info("🏥 Simple Explainability Toolkit Starting")
    logger.info("=" * 50)

    try:
        explainer = SimpleExplainer()
        results = explainer.explain_model(args.model_path, args.model_name)

        print(f"\n🎉 EXPLANATION COMPLETE")
        print("=" * 30)
        print(f"Model: {results['model_name']}")
        print(f"Methods: {len(results['importance_results'])}")
        print(f"Visualizations: {len(results['visualizations'])}")
        print(f"Report: {results['report_path']}")

        methods_used = list(results["importance_results"].keys())
        print(f"Explanation methods: {', '.join(methods_used).upper()}")

        return 0

    except Exception as e:
        logger.error(f"❌ Explanation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
