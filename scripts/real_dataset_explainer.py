"""
Real Dataset Explainer

Specialized explainer for models trained on real healthcare datasets.
"""

import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import logging
from sklearn.inspection import permutation_importance
import shap
import matplotlib.pyplot as plt

# Add scripts to path
sys.path.append(str(Path(__file__).parent))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealDatasetExplainer:
    """Explainer for real healthcare dataset models."""

    def __init__(self, output_dir: str = "reports/real_explanations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"🏥 Real Dataset Explainer initialized - Output: {self.output_dir}"
        )

    def load_model_data(self, model_path: str):
        """Load model and associated data from joblib file."""
        try:
            model_data = joblib.load(model_path)

            # Extract components
            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.feature_names = model_data["feature_names"]
            self.X_test = model_data["X_test"]
            self.y_test = model_data["y_test"]
            self.dataset_name = model_data.get("dataset_name", "unknown")
            self.accuracy = model_data.get("accuracy", 0.0)

            logger.info(f"✅ Model loaded: {self.dataset_name}")
            logger.info(f"   Model type: {type(self.model).__name__}")
            logger.info(f"   Features: {len(self.feature_names)}")
            logger.info(f"   Test samples: {len(self.X_test)}")
            logger.info(f"   Accuracy: {self.accuracy:.3f}")

            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def explain_model(self, model_path: str, model_name: str):
        """Generate comprehensive explanations for the model."""
        if not self.load_model_data(model_path):
            return None

        results = {
            "model_name": model_name,
            "dataset_name": self.dataset_name,
            "model_type": type(self.model).__name__,
            "accuracy": self.accuracy,
            "feature_names": self.feature_names,
            "explanations": {},
        }

        # 1. Built-in feature importance (if available)
        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
            results["explanations"]["builtin"] = {
                "importance": importance.tolist(),
                "feature_names": self.feature_names,
            }
            logger.info("✅ Built-in feature importance extracted")

        # 2. Permutation importance
        try:
            perm_importance = permutation_importance(
                self.model, self.X_test, self.y_test, n_repeats=10, random_state=42
            )

            results["explanations"]["permutation"] = {
                "importance": perm_importance.importances_mean.tolist(),
                "importance_std": perm_importance.importances_std.tolist(),
                "feature_names": self.feature_names,
            }
            logger.info("✅ Permutation importance calculated")

        except Exception as e:
            logger.error(f"Permutation importance failed: {e}")

        # 3. SHAP explanations
        try:
            # Use a subset of test data for SHAP
            sample_size = min(50, len(self.X_test))
            X_sample = self.X_test.sample(n=sample_size, random_state=42)

            if hasattr(self.model, "predict_proba"):
                # For tree-based models
                if hasattr(self.model, "estimators_"):
                    explainer = shap.TreeExplainer(self.model)
                else:
                    # For other models, use Kernel explainer with background data
                    background = shap.sample(self.X_test, 20)
                    explainer = shap.KernelExplainer(
                        self.model.predict_proba, background
                    )
            else:
                # For models without predict_proba
                background = shap.sample(self.X_test, 20)
                explainer = shap.KernelExplainer(self.model.predict, background)

            shap_values = explainer.shap_values(X_sample)

            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

            # Ensure 2D array
            if len(shap_values.shape) > 2:
                shap_values = shap_values[:, :, 0]

            feature_importance = np.abs(shap_values).mean(axis=0)

            results["explanations"]["shap"] = {
                "values": shap_values.tolist(),
                "feature_importance": feature_importance.tolist(),
                "feature_names": self.feature_names,
                "expected_value": (
                    float(explainer.expected_value)
                    if hasattr(explainer, "expected_value")
                    else 0.0
                ),
            }
            logger.info("✅ SHAP explanations calculated")

        except Exception as e:
            logger.error(f"SHAP calculation failed: {e}")

        # 4. Generate visualizations
        self.create_visualizations(results, model_name)

        # 5. Clinical interpretation
        clinical_insights = self.generate_clinical_insights(results)
        results["clinical_insights"] = clinical_insights

        # Save results
        report_path = self.output_dir / f"{model_name}_explanation_report.json"
        import json

        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"📋 Explanation report saved: {report_path}")
        return results

    def create_visualizations(self, results, model_name):
        """Create explanation visualizations."""
        explanations = results["explanations"]

        # Count available methods
        n_methods = len(explanations)
        if n_methods == 0:
            return

        # Create subplot grid
        fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 6))
        if n_methods == 1:
            axes = [axes]

        fig.suptitle(
            f"{model_name} - Feature Importance Analysis\n"
            f"Dataset: {self.dataset_name} | Accuracy: {self.accuracy:.3f}",
            fontsize=14,
        )

        method_idx = 0

        # Built-in importance
        if "builtin" in explanations:
            data = explanations["builtin"]
            importance = np.array(data["importance"])
            feature_names = data["feature_names"]

            # Sort by importance
            sorted_idx = np.argsort(importance)

            axes[method_idx].barh(range(len(feature_names)), importance[sorted_idx])
            axes[method_idx].set_yticks(range(len(feature_names)))
            axes[method_idx].set_yticklabels([feature_names[i] for i in sorted_idx])
            axes[method_idx].set_xlabel("Feature Importance")
            axes[method_idx].set_title("Built-in Feature Importance")
            axes[method_idx].grid(True, alpha=0.3)

            method_idx += 1

        # Permutation importance
        if "permutation" in explanations:
            data = explanations["permutation"]
            importance = np.array(data["importance"])
            feature_names = data["feature_names"]

            sorted_idx = np.argsort(importance)

            axes[method_idx].barh(range(len(feature_names)), importance[sorted_idx])
            axes[method_idx].set_yticks(range(len(feature_names)))
            axes[method_idx].set_yticklabels([feature_names[i] for i in sorted_idx])
            axes[method_idx].set_xlabel("Permutation Importance")
            axes[method_idx].set_title("Permutation Feature Importance")
            axes[method_idx].grid(True, alpha=0.3)

            method_idx += 1

        # SHAP importance
        if "shap" in explanations:
            data = explanations["shap"]
            importance = np.array(data["feature_importance"])
            feature_names = data["feature_names"]

            sorted_idx = np.argsort(importance)

            axes[method_idx].barh(range(len(feature_names)), importance[sorted_idx])
            axes[method_idx].set_yticks(range(len(feature_names)))
            axes[method_idx].set_yticklabels([feature_names[i] for i in sorted_idx])
            axes[method_idx].set_xlabel("SHAP Importance")
            axes[method_idx].set_title("SHAP Feature Importance")
            axes[method_idx].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / f"{model_name}_feature_importance.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"📊 Visualization saved: {plot_path}")

    def generate_clinical_insights(self, results):
        """Generate clinical insights based on feature importance."""
        explanations = results["explanations"]
        dataset_name = results["dataset_name"]

        insights = {
            "dataset": dataset_name,
            "key_findings": [],
            "clinical_recommendations": [],
        }

        # Get top features from available explanations
        top_features = []

        for method in ["builtin", "permutation", "shap"]:
            if method in explanations:
                data = explanations[method]
                if method == "shap":
                    importance = data["feature_importance"]
                else:
                    importance = data["importance"]

                feature_names = data["feature_names"]

                # Get top 5 features
                sorted_idx = np.argsort(importance)[-5:][::-1]
                method_top_features = [
                    (feature_names[i], importance[i]) for i in sorted_idx
                ]
                top_features.extend(method_top_features)

        # Remove duplicates and get overall top features
        feature_importance_dict = {}
        for feature, importance in top_features:
            if feature in feature_importance_dict:
                feature_importance_dict[feature].append(importance)
            else:
                feature_importance_dict[feature] = [importance]

        # Average importance across methods
        avg_importance = {
            feature: np.mean(importances)
            for feature, importances in feature_importance_dict.items()
        }

        # Sort by average importance
        top_clinical_features = sorted(
            avg_importance.items(), key=lambda x: x[1], reverse=True
        )[:5]

        # Generate insights based on dataset and top features
        if dataset_name == "heart_disease_uci":
            insights["key_findings"] = [
                f"Most predictive feature: {top_clinical_features[0][0]} (importance: {top_clinical_features[0][1]:.3f})",
                f"Top cardiovascular risk factors: {', '.join([f[0] for f in top_clinical_features[:3]])}",
            ]

            clinical_recs = []
            for feature, importance in top_clinical_features:
                if "cp" in feature:  # chest pain
                    clinical_recs.append(
                        "Chest pain type is highly predictive - detailed pain assessment crucial"
                    )
                elif "thalach" in feature:  # max heart rate
                    clinical_recs.append(
                        "Maximum heart rate achieved during exercise is significant"
                    )
                elif "oldpeak" in feature:  # ST depression
                    clinical_recs.append(
                        "ST depression (oldpeak) indicates exercise-induced ischemia"
                    )
                elif "ca" in feature:  # coronary arteries
                    clinical_recs.append(
                        "Number of major vessels colored by fluoroscopy is important"
                    )
                elif "thal" in feature:  # thalassemia
                    clinical_recs.append(
                        "Thalassemia status affects cardiac risk assessment"
                    )

            insights["clinical_recommendations"] = clinical_recs

        elif dataset_name == "breast_cancer_wisconsin":
            insights["key_findings"] = [
                f"Most predictive feature: {top_clinical_features[0][0]} (importance: {top_clinical_features[0][1]:.3f})",
                f"Key diagnostic features: {', '.join([f[0] for f in top_clinical_features[:3]])}",
            ]

            insights["clinical_recommendations"] = [
                "Focus on morphological features with highest predictive value",
                "Consider texture and area measurements for diagnostic accuracy",
                "Worst-case features often more predictive than mean values",
            ]

        elif dataset_name == "diabetes_130_hospitals":
            insights["key_findings"] = [
                f"Most predictive factor: {top_clinical_features[0][0]} (importance: {top_clinical_features[0][1]:.3f})",
                f"Key readmission predictors: {', '.join([f[0] for f in top_clinical_features[:3]])}",
            ]

            insights["clinical_recommendations"] = [
                "Monitor patients with high number of previous emergency visits",
                "Consider medication count as readmission risk factor",
                "Age and length of stay are significant predictors",
            ]

        else:
            # Generic insights
            insights["key_findings"] = [
                f"Most important feature: {top_clinical_features[0][0]}",
                f"Top predictive features: {', '.join([f[0] for f in top_clinical_features[:3]])}",
            ]

            insights["clinical_recommendations"] = [
                "Focus clinical attention on high-importance features",
                "Consider feature interactions in clinical decision-making",
            ]

        return insights


def main():
    """Example usage and testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Real Dataset Model Explainer")
    parser.add_argument(
        "--model-path", required=True, help="Path to trained model joblib file"
    )
    parser.add_argument("--model-name", required=True, help="Name for the model")

    args = parser.parse_args()

    # Initialize explainer
    explainer = RealDatasetExplainer()

    # Generate explanations
    results = explainer.explain_model(args.model_path, args.model_name)

    if results:
        logger.info(f"\n🎉 EXPLANATION COMPLETE")
        logger.info(f"Model: {results['model_name']}")
        logger.info(f"Dataset: {results['dataset_name']}")
        logger.info(f"Accuracy: {results['accuracy']:.3f}")
        logger.info(f"Explanation methods: {len(results['explanations'])}")
        logger.info(f"Reports saved to: {explainer.output_dir}")

        # Print top features
        if "builtin" in results["explanations"]:
            builtin = results["explanations"]["builtin"]
            importance = np.array(builtin["importance"])
            feature_names = builtin["feature_names"]
            top_idx = np.argsort(importance)[-5:][::-1]

            logger.info(f"\n🏆 Top 5 Features (Built-in Importance):")
            for i, idx in enumerate(top_idx, 1):
                logger.info(f"  {i}. {feature_names[idx]}: {importance[idx]:.4f}")


if __name__ == "__main__":
    main()
