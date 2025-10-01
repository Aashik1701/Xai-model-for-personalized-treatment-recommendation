#!/usr/bin/env python3
"""
Explainability Toolkit - Comprehensive Model Explanation System

This toolkit provides SHAP and LIME explanations for healthcare ML models with:
- Multiple explanation methods (SHAP, LIME, Permutation Importance)
- Clinical interpretation and risk factor analysis
- Interactive visualizations and reports
- Production-ready explanation APIs
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
import json
import warnings

warnings.filterwarnings("ignore")

# ML and explanation libraries
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer

    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

# Project imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mlflow_manager import MLflowManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("explainability_toolkit")


class ExplainabilityToolkit:
    """Comprehensive explainability toolkit for healthcare ML models."""

    def __init__(self, config_path: str = "config/explainability_config.yaml"):
        """Initialize the explainability toolkit."""
        self.config_path = config_path
        self.config = self._load_config()

        # Initialize directories
        self.output_dir = Path(self.config["output"]["base_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize MLflow manager
        self.mlflow_manager = MLflowManager(experiment_name="explainability-analysis")

        # Check library availability
        self._check_dependencies()

        logger.info("Explainability Toolkit initialized")
        logger.info(f"Output directory: {self.output_dir}")

    def _load_config(self) -> Dict[str, Any]:
        """Load explainability configuration."""
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "shap": {
                "explanations": {"max_evals": 500, "batch_size": 50},
                "background_data": {"sample_size": 100, "strategy": "random"},
            },
            "lime": {"explanations": {"num_features": 10, "num_samples": 1000}},
            "output": {"formats": ["json", "png"], "base_dir": "reports/explanations"},
            "performance": {"n_jobs": -1, "cache_explanations": True},
        }

    def _check_dependencies(self):
        """Check if required libraries are available."""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Install with: pip install shap")
        if not LIME_AVAILABLE:
            logger.warning("LIME not available. Install with: pip install lime")

        if not (SHAP_AVAILABLE or LIME_AVAILABLE):
            raise ImportError("Neither SHAP nor LIME available. Install at least one.")

    def load_model_and_data(
        self, model_path: str
    ) -> Tuple[Any, pd.DataFrame, pd.Series]:
        """Load trained model and generate sample data for explanations."""
        logger.info(f"Loading model from {model_path}")

        # Load model
        model = joblib.load(model_path)

        # Generate sample data (same as training for consistency)
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

        # Feature names for healthcare context
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

        # Use same split as training
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_series, test_size=0.2, random_state=42, stratify=y_series
        )

        # Scale data (assuming model was trained on scaled data)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )

        return model, X_test_scaled, y_test

    def create_shap_explainer(
        self, model: Any, X_background: pd.DataFrame, model_type: str = "auto"
    ) -> Any:
        """Create appropriate SHAP explainer for the model."""
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not available")

        logger.info(f"Creating SHAP explainer for {model_type}")

        # Determine explainer type
        if model_type == "auto":
            model_name = model.__class__.__name__.lower()
            if "forest" in model_name or "tree" in model_name:
                model_type = "tree"
            elif "linear" in model_name or "logistic" in model_name:
                model_type = "linear"
            else:
                model_type = "kernel"

        # Create explainer
        if model_type == "tree" and hasattr(model, "estimators_"):
            explainer = shap.TreeExplainer(model)
        elif model_type == "linear":
            explainer = shap.LinearExplainer(model, X_background)
        else:
            # Use kernel explainer as fallback
            background_sample = shap.sample(X_background, 100)
            explainer = shap.KernelExplainer(model.predict_proba, background_sample)

        return explainer

    def create_lime_explainer(
        self, X_train: pd.DataFrame, mode: str = "classification"
    ) -> Any:
        """Create LIME tabular explainer."""
        if not LIME_AVAILABLE:
            raise ImportError("LIME not available")

        logger.info("Creating LIME explainer")

        lime_config = self.config.get("lime", {})
        tabular_config = lime_config.get("tabular", {})

        explainer = LimeTabularExplainer(
            X_train.values,
            feature_names=X_train.columns.tolist(),
            class_names=["Low Risk", "High Risk"],
            mode=mode,
            discretize_continuous=tabular_config.get("discretize_continuous", True),
            discretizer=tabular_config.get("discretizer", "quartiles"),
        )

        return explainer

    def generate_shap_explanations(
        self, explainer: Any, X_test: pd.DataFrame, sample_size: int = 50
    ) -> Dict[str, Any]:
        """Generate SHAP explanations for test samples."""
        logger.info(f"Generating SHAP explanations for {sample_size} samples")

        # Sample data if needed
        if len(X_test) > sample_size:
            X_sample = X_test.sample(n=sample_size, random_state=42)
        else:
            X_sample = X_test

        # Generate SHAP values
        try:
            if hasattr(explainer, "shap_values"):
                shap_values = explainer.shap_values(X_sample)
                # Handle multi-class output
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Use positive class
            else:
                shap_values = explainer(X_sample).values
                if len(shap_values.shape) == 3:
                    shap_values = shap_values[:, :, 1]  # Use positive class

        except Exception as e:
            logger.error(f"Error generating SHAP values: {e}")
            return {}

        # Calculate feature importance
        feature_importance = np.abs(shap_values).mean(axis=0)

        return {
            "shap_values": shap_values,
            "feature_importance": feature_importance,
            "feature_names": X_sample.columns.tolist(),
            "samples": X_sample,
            "expected_value": getattr(explainer, "expected_value", 0),
        }

    def generate_lime_explanations(
        self, explainer: Any, model: Any, X_test: pd.DataFrame, sample_size: int = 10
    ) -> Dict[str, Any]:
        """Generate LIME explanations for test samples."""
        logger.info(f"Generating LIME explanations for {sample_size} samples")

        # Sample data
        if len(X_test) > sample_size:
            X_sample = X_test.sample(n=sample_size, random_state=42)
        else:
            X_sample = X_test

        lime_config = self.config.get("lime", {})
        exp_config = lime_config.get("explanations", {})

        explanations = []
        feature_importance_all = []

        for idx, (_, row) in enumerate(X_sample.iterrows()):
            try:
                # Generate explanation
                exp = explainer.explain_instance(
                    row.values,
                    model.predict_proba,
                    num_features=exp_config.get("num_features", 10),
                    num_samples=exp_config.get("num_samples", 1000),
                )

                # Extract feature importance
                feature_importance = dict(exp.as_list())
                feature_importance_all.append(feature_importance)

                explanations.append(
                    {
                        "instance_id": idx,
                        "prediction_proba": model.predict_proba([row.values])[0],
                        "explanation": exp.as_list(),
                        "score": exp.score,
                    }
                )

            except Exception as e:
                logger.error(f"Error generating LIME explanation for sample {idx}: {e}")
                continue

        return {
            "explanations": explanations,
            "feature_importance_all": feature_importance_all,
            "feature_names": X_sample.columns.tolist(),
            "samples": X_sample,
        }

    def calculate_permutation_importance(
        self, model: Any, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, Any]:
        """Calculate permutation feature importance."""
        logger.info("Calculating permutation importance")

        perf_config = self.config.get("performance", {})

        try:
            perm_importance = permutation_importance(
                model,
                X_test,
                y_test,
                n_repeats=10,
                random_state=42,
                n_jobs=perf_config.get("n_jobs", -1),
                scoring="roc_auc",
            )

            return {
                "importances_mean": perm_importance.importances_mean,
                "importances_std": perm_importance.importances_std,
                "feature_names": X_test.columns.tolist(),
            }

        except Exception as e:
            logger.error(f"Error calculating permutation importance: {e}")
            return {}

    def create_explanation_visualizations(
        self, explanations: Dict[str, Any], model_name: str
    ) -> List[str]:
        """Create comprehensive explanation visualizations."""
        logger.info("Creating explanation visualizations")

        saved_plots = []

        # SHAP visualizations
        if "shap_values" in explanations:
            shap_data = explanations["shap_values"]
            saved_plots.extend(self._create_shap_plots(shap_data, model_name))

        # Feature importance comparison
        if any(key in explanations for key in ["shap_values", "permutation"]):
            saved_plots.append(
                self._create_feature_importance_comparison(explanations, model_name)
            )

        # Clinical interpretation
        saved_plots.append(
            self._create_clinical_interpretation(explanations, model_name)
        )

        return saved_plots

    def _create_shap_plots(
        self, shap_data: Dict[str, Any], model_name: str
    ) -> List[str]:
        """Create SHAP-specific visualizations."""
        saved_plots = []

        try:
            shap_values = shap_data["shap_values"]
            samples = shap_data["samples"]

            # Summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values, samples, plot_type="dot", show=False, max_display=15
            )
            plt.title(f"{model_name} - SHAP Feature Importance", fontsize=16)
            summary_path = self.output_dir / f"{model_name}_shap_summary.png"
            plt.savefig(summary_path, dpi=300, bbox_inches="tight")
            plt.close()
            saved_plots.append(str(summary_path))

            # Bar plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values, samples, plot_type="bar", show=False, max_display=15
            )
            plt.title(f"{model_name} - SHAP Feature Importance (Bar)", fontsize=16)
            bar_path = self.output_dir / f"{model_name}_shap_bar.png"
            plt.savefig(bar_path, dpi=300, bbox_inches="tight")
            plt.close()
            saved_plots.append(str(bar_path))

            # Waterfall plot for first sample
            if len(shap_values) > 0:
                plt.figure(figsize=(10, 8))
                expected_value = shap_data.get("expected_value", 0)
                if hasattr(expected_value, "__iter__") and len(expected_value) > 1:
                    expected_value = expected_value[1]  # Positive class

                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_values[0],
                        base_values=expected_value,
                        data=samples.iloc[0],
                        feature_names=samples.columns.tolist(),
                    ),
                    show=False,
                )
                plt.title(f"{model_name} - SHAP Waterfall (Sample 1)", fontsize=16)
                waterfall_path = self.output_dir / f"{model_name}_shap_waterfall.png"
                plt.savefig(waterfall_path, dpi=300, bbox_inches="tight")
                plt.close()
                saved_plots.append(str(waterfall_path))

        except Exception as e:
            logger.error(f"Error creating SHAP plots: {e}")

        return saved_plots

    def _create_feature_importance_comparison(
        self, explanations: Dict[str, Any], model_name: str
    ) -> str:
        """Create feature importance comparison plot."""
        plt.figure(figsize=(14, 8))

        importance_data = {}

        # SHAP importance
        if "shap_values" in explanations:
            shap_importance = explanations["shap_values"]["feature_importance"]
            feature_names = explanations["shap_values"]["feature_names"]
            importance_data["SHAP"] = dict(zip(feature_names, shap_importance))

        # Permutation importance
        if "permutation" in explanations:
            perm_importance = explanations["permutation"]["importances_mean"]
            feature_names = explanations["permutation"]["feature_names"]
            importance_data["Permutation"] = dict(zip(feature_names, perm_importance))

        # Built-in importance (if available)
        if "builtin" in explanations:
            builtin_importance = explanations["builtin"]["importance"]
            feature_names = explanations["builtin"]["feature_names"]
            importance_data["Built-in"] = dict(zip(feature_names, builtin_importance))

        if importance_data:
            # Create comparison DataFrame
            comparison_df = pd.DataFrame(importance_data).fillna(0)

            # Plot comparison
            ax = comparison_df.plot(kind="bar", figsize=(14, 8), rot=45)
            ax.set_title(f"{model_name} - Feature Importance Comparison", fontsize=16)
            ax.set_xlabel("Features", fontsize=12)
            ax.set_ylabel("Importance Score", fontsize=12)
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()

            comparison_path = (
                self.output_dir / f"{model_name}_importance_comparison.png"
            )
            plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
            plt.close()

            return str(comparison_path)

        return ""

    def _create_clinical_interpretation(
        self, explanations: Dict[str, Any], model_name: str
    ) -> str:
        """Create clinical interpretation visualization."""
        plt.figure(figsize=(14, 10))

        # Get feature importance from any available method
        feature_importance = {}
        if "shap_values" in explanations:
            shap_data = explanations["shap_values"]
            feature_importance = dict(
                zip(shap_data["feature_names"], shap_data["feature_importance"])
            )
        elif "permutation" in explanations:
            perm_data = explanations["permutation"]
            feature_importance = dict(
                zip(perm_data["feature_names"], perm_data["importances_mean"])
            )

        if not feature_importance:
            return ""

        # Group features by clinical category
        clinical_groups = self.config.get("clinical", {}).get("feature_groups", {})

        grouped_importance = {}
        for group_name, features in clinical_groups.items():
            group_importance = sum(
                feature_importance.get(feature, 0)
                for feature in features
                if feature in feature_importance
            )
            if group_importance > 0:
                grouped_importance[group_name] = group_importance

        # Ungrouped features
        grouped_features = set()
        for features in clinical_groups.values():
            grouped_features.update(features)

        ungrouped = {
            feature: importance
            for feature, importance in feature_importance.items()
            if feature not in grouped_features
        }

        if ungrouped:
            grouped_importance["Other"] = sum(ungrouped.values())

        # Create clinical interpretation plot
        if grouped_importance:
            # Pie chart of clinical groups
            plt.subplot(2, 2, 1)
            plt.pie(
                grouped_importance.values(),
                labels=grouped_importance.keys(),
                autopct="%1.1f%%",
                startangle=90,
            )
            plt.title("Importance by Clinical Category")

            # Bar plot of top individual features
            plt.subplot(2, 2, 2)
            sorted_features = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:10]

            features, importances = zip(*sorted_features)
            plt.barh(range(len(features)), importances)
            plt.yticks(range(len(features)), features)
            plt.xlabel("Feature Importance")
            plt.title("Top 10 Individual Features")
            plt.gca().invert_yaxis()

            # Risk factor analysis
            plt.subplot(2, 1, 2)
            risk_analysis = self._analyze_risk_factors(feature_importance)

            plt.text(
                0.1,
                0.8,
                "Risk Factor Analysis:",
                fontsize=14,
                fontweight="bold",
                transform=plt.gca().transAxes,
            )

            y_pos = 0.6
            for category, analysis in risk_analysis.items():
                plt.text(
                    0.1,
                    y_pos,
                    f"• {category}: {analysis}",
                    fontsize=11,
                    transform=plt.gca().transAxes,
                    wrap=True,
                )
                y_pos -= 0.15

            plt.axis("off")

            plt.suptitle(f"{model_name} - Clinical Interpretation", fontsize=16)
            plt.tight_layout()

            clinical_path = (
                self.output_dir / f"{model_name}_clinical_interpretation.png"
            )
            plt.savefig(clinical_path, dpi=300, bbox_inches="tight")
            plt.close()

            return str(clinical_path)

        return ""

    def _analyze_risk_factors(
        self, feature_importance: Dict[str, float]
    ) -> Dict[str, str]:
        """Analyze risk factors based on feature importance."""
        analysis = {}

        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )

        if len(sorted_features) >= 3:
            top_features = [f[0] for f in sorted_features[:3]]
            analysis["Primary Risk Factors"] = (
                f"Most influential: {', '.join(top_features)}"
            )

        # Clinical category analysis
        clinical_groups = self.config.get("clinical", {}).get("feature_groups", {})
        group_importance = {}

        for group_name, features in clinical_groups.items():
            total_importance = sum(
                feature_importance.get(feature, 0) for feature in features
            )
            if total_importance > 0:
                group_importance[group_name] = total_importance

        if group_importance:
            top_group = max(group_importance.items(), key=lambda x: x[1])
            analysis["Key Clinical Domain"] = f"{top_group[0]} factors most important"

        # Modifiable vs non-modifiable
        modifiable = ["smoking", "exercise", "diet_quality", "stress_level", "bmi"]
        non_modifiable = ["age", "sex", "family_history"]

        modifiable_importance = sum(
            feature_importance.get(feature, 0) for feature in modifiable
        )
        non_modifiable_importance = sum(
            feature_importance.get(feature, 0) for feature in non_modifiable
        )

        if modifiable_importance > non_modifiable_importance:
            analysis["Intervention Potential"] = "High - many modifiable risk factors"
        else:
            analysis["Intervention Potential"] = "Moderate - focus on lifestyle changes"

        return analysis

    def generate_explanation_report(
        self, model_name: str, explanations: Dict[str, Any], visualizations: List[str]
    ) -> str:
        """Generate comprehensive explanation report."""
        logger.info(f"Generating explanation report for {model_name}")

        report = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "explanation_methods": list(explanations.keys()),
            "visualizations": visualizations,
            "summary": {},
        }

        # Feature importance summary
        if "shap_values" in explanations:
            shap_data = explanations["shap_values"]
            top_features = sorted(
                zip(shap_data["feature_names"], shap_data["feature_importance"]),
                key=lambda x: x[1],
                reverse=True,
            )[:5]

            report["summary"]["top_shap_features"] = [
                {"feature": feature, "importance": float(importance)}
                for feature, importance in top_features
            ]

        # Clinical interpretation
        if "shap_values" in explanations or "permutation" in explanations:
            feature_importance = {}
            if "shap_values" in explanations:
                shap_data = explanations["shap_values"]
                feature_importance = dict(
                    zip(shap_data["feature_names"], shap_data["feature_importance"])
                )

            risk_analysis = self._analyze_risk_factors(feature_importance)
            report["summary"]["clinical_analysis"] = risk_analysis

        # Save report
        report_path = self.output_dir / f"{model_name}_explanation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Explanation report saved: {report_path}")
        return str(report_path)

    def explain_model(self, model_path: str, model_name: str = None) -> Dict[str, Any]:
        """Generate comprehensive explanations for a trained model."""
        if model_name is None:
            model_name = Path(model_path).stem.split("_")[0]

        logger.info(f"🔍 Starting explanation analysis for {model_name}")

        # Load model and data
        model, X_test, y_test = self.load_model_and_data(model_path)

        # Generate background data for SHAP
        X_background = X_test.sample(n=min(100, len(X_test)), random_state=42)

        explanations = {}

        # SHAP explanations
        if SHAP_AVAILABLE:
            try:
                logger.info("Generating SHAP explanations...")
                explainer = self.create_shap_explainer(model, X_background)
                shap_results = self.generate_shap_explanations(explainer, X_test)
                if shap_results:
                    explanations["shap_values"] = shap_results

            except Exception as e:
                logger.error(f"SHAP explanation failed: {e}")

        # LIME explanations
        if LIME_AVAILABLE:
            try:
                logger.info("Generating LIME explanations...")
                lime_explainer = self.create_lime_explainer(X_background)
                lime_results = self.generate_lime_explanations(
                    lime_explainer, model, X_test, sample_size=5
                )
                if lime_results:
                    explanations["lime"] = lime_results

            except Exception as e:
                logger.error(f"LIME explanation failed: {e}")

        # Permutation importance
        try:
            logger.info("Calculating permutation importance...")
            perm_results = self.calculate_permutation_importance(model, X_test, y_test)
            if perm_results:
                explanations["permutation"] = perm_results

        except Exception as e:
            logger.error(f"Permutation importance failed: {e}")

        # Built-in feature importance (for tree models)
        if hasattr(model, "feature_importances_"):
            explanations["builtin"] = {
                "importance": model.feature_importances_,
                "feature_names": X_test.columns.tolist(),
            }

        # Create visualizations
        visualizations = self.create_explanation_visualizations(
            explanations, model_name
        )

        # Generate report
        report_path = self.generate_explanation_report(
            model_name, explanations, visualizations
        )

        # Log to MLflow
        with self.mlflow_manager.start_run(run_name=f"{model_name}_explanations"):
            # Log explanation metrics
            metrics = {}
            if "shap_values" in explanations:
                metrics["shap_features_analyzed"] = len(
                    explanations["shap_values"]["feature_names"]
                )
            if "lime" in explanations:
                metrics["lime_samples_explained"] = len(
                    explanations["lime"]["explanations"]
                )

            self.mlflow_manager.log_model_metrics(metrics, prefix="explanation")

        logger.info(f"✅ Explanation analysis complete for {model_name}")
        logger.info(f"📊 Report: {report_path}")
        logger.info(f"🎨 Visualizations: {len(visualizations)} plots created")

        return {
            "model_name": model_name,
            "explanations": explanations,
            "visualizations": visualizations,
            "report_path": report_path,
        }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Explainability Toolkit for Healthcare ML Models"
    )
    parser.add_argument(
        "--model-path", required=True, help="Path to trained model file"
    )
    parser.add_argument("--model-name", help="Model name for reporting")
    parser.add_argument(
        "--config",
        default="config/explainability_config.yaml",
        help="Configuration file path",
    )

    args = parser.parse_args()

    logger.info("🏥 Starting Healthcare Explainability Toolkit")
    logger.info("=" * 60)

    try:
        # Initialize toolkit
        toolkit = ExplainabilityToolkit(args.config)

        # Generate explanations
        results = toolkit.explain_model(args.model_path, args.model_name)

        # Print summary
        print(f"\n🎉 EXPLANATION ANALYSIS COMPLETE")
        print("=" * 40)
        print(f"Model: {results['model_name']}")
        print(f"Explanation methods: {len(results['explanations'])}")
        print(f"Visualizations: {len(results['visualizations'])}")
        print(f"Report: {results['report_path']}")

        if results["explanations"]:
            print("\nAvailable explanations:")
            for method in results["explanations"].keys():
                print(f"  • {method.upper()}")

        logger.info("✅ Explainability Toolkit execution complete!")

    except Exception as e:
        logger.error(f"❌ Explainability analysis failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
