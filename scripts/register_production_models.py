#!/usr/bin/env python3
"""
Register Top-Pe    {
        "name": "BreastCancer",
        "run_name": "breast_cancer_wisconsin_RandomForest",
        "description": "Breast cancer diagnosis (97.37% accuracy)",
        "use_case": "Cancer diagnostic support and treatment planning"
    },
    {
        "name": "HeartDisease",
        "run_name": "heart_disease_uci_RandomForest",
        "description": "Cardiovascular disease prediction (88.52% accuracy)",
        "use_case": "Cardiac risk assessment and preventive care"
    }els to MLflow Model Registry

This script registers the 5 best-performing models from our training runs
to the MLflow Model Registry and tags them for production deployment.

Usage:
    python scripts/register_production_models.py
"""

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import os
from pathlib import Path
from datetime import datetime

# MLflow setup
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Top 5 models to register (based on training results)
TOP_MODELS = [
    {
        "name": "SyntheaLongitudinal",
        "run_name": "synthea_longitudinal_RandomForest",
        "description": "Longitudinal EHR model for patient risk assessment (100% accuracy)",
        "use_case": "Temporal patient risk prediction with multi-encounter history",
    },
    {
        "name": "ISICSkinCancer",
        "run_name": "skin_cancer_imaging_RandomForest",
        "description": "Skin cancer detection from imaging metadata (100% accuracy)",
        "use_case": "Dermatology screening and cancer detection",
    },
    {
        "name": "DrugReviews",
        "run_name": "drug_reviews_GradientBoosting",
        "description": "Drug effectiveness prediction from patient reviews (99.88% accuracy)",
        "use_case": "Treatment effectiveness assessment and drug recommendation",
    },
    {
        "name": "BreastCancer",
        "run_name": "breast_cancer_wisconsin_RandomForest",
        "description": "Breast cancer diagnosis from diagnostic features (97.37% accuracy)",
        "use_case": "Cancer diagnostic support and treatment planning",
    },
    {
        "name": "HeartDisease",
        "run_name": "heart_disease_uci_RandomForest",
        "description": "Cardiovascular disease prediction (88.52% accuracy)",
        "use_case": "Cardiac risk assessment and preventive care",
    },
]


def find_run_by_name(experiment_name: str, run_name: str):
    """Find MLflow run by name within an experiment"""
    client = MlflowClient()

    # Get experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    # Search for run
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
    )

    if not runs:
        # Try searching by run name in params
        all_runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        for run in all_runs:
            dataset = run.data.params.get("dataset", "")
            model = run.data.params.get("model", "")
            constructed_name = f"{dataset}_{model}"
            if constructed_name == run_name:
                return run

        raise ValueError(
            f"Run '{run_name}' not found in experiment '{experiment_name}'"
        )

    return runs[0]


def register_model(model_info: dict, experiment_name: str):
    """Register a model to MLflow Model Registry"""
    client = MlflowClient()

    print(f"\n{'='*80}")
    print(f"Registering: {model_info['name']}")
    print(f"{'='*80}")

    try:
        # Find the run
        run = find_run_by_name(experiment_name, model_info["run_name"])
        run_id = run.info.run_id

        print(f"✓ Found run: {run_id}")

        # Get run metrics
        accuracy = run.data.metrics.get("accuracy", 0)
        f1_score = run.data.metrics.get("f1_score", 0)
        precision = run.data.metrics.get("precision", 0)
        recall = run.data.metrics.get("recall", 0)

        print(f"  - Accuracy: {accuracy:.4f}")
        print(f"  - F1 Score: {f1_score:.4f}")
        print(f"  - Precision: {precision:.4f}")
        print(f"  - Recall: {recall:.4f}")

        # Register model
        model_uri = f"runs:/{run_id}/model"

        try:
            # Check if model already exists
            client.get_registered_model(model_info["name"])
            print(f"  Model '{model_info['name']}' already exists in registry")

            # Create new version
            model_version = client.create_model_version(
                name=model_info["name"],
                source=model_uri,
                run_id=run_id,
                description=f"Version created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            )
            print(f"✓ Created new version: {model_version.version}")

        except mlflow.exceptions.RestException:
            # Model doesn't exist, create it
            client.create_registered_model(
                name=model_info["name"], description=model_info["description"]
            )
            print(f"✓ Created new model in registry: {model_info['name']}")

            # Create first version
            model_version = client.create_model_version(
                name=model_info["name"],
                source=model_uri,
                run_id=run_id,
                description="Initial production version",
            )
            print(f"✓ Created version: {model_version.version}")

        # Add tags
        client.set_model_version_tag(
            name=model_info["name"],
            version=model_version.version,
            key="use_case",
            value=model_info["use_case"],
        )

        client.set_model_version_tag(
            name=model_info["name"],
            version=model_version.version,
            key="accuracy",
            value=str(accuracy),
        )

        client.set_model_version_tag(
            name=model_info["name"],
            version=model_version.version,
            key="deployment_date",
            value=datetime.now().strftime("%Y-%m-%d"),
        )

        print(f"✓ Added tags to version")

        # Transition to Production
        client.transition_model_version_stage(
            name=model_info["name"],
            version=model_version.version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"✓ Transitioned to Production stage")

        # Update model description
        client.update_registered_model(
            name=model_info["name"],
            description=f"{model_info['description']}\n\nUse Case: {model_info['use_case']}",
        )

        print(
            f"✅ Successfully registered {model_info['name']} (v{model_version.version})"
        )

        return {
            "model_name": model_info["name"],
            "version": model_version.version,
            "run_id": run_id,
            "accuracy": accuracy,
            "status": "success",
        }

    except Exception as e:
        print(f"❌ Error registering {model_info['name']}: {str(e)}")
        return {
            "model_name": model_info["name"],
            "version": None,
            "run_id": None,
            "accuracy": None,
            "status": "failed",
            "error": str(e),
        }


def generate_registry_report(results: list):
    """Generate a summary report of registered models"""
    print(f"\n{'='*80}")
    print("MODEL REGISTRY SUMMARY")
    print(f"{'='*80}\n")

    df = pd.DataFrame(results)

    successful = df[df["status"] == "success"]
    failed = df[df["status"] == "failed"]

    print(f"✅ Successfully Registered: {len(successful)}/{len(results)}")
    print(f"❌ Failed: {len(failed)}/{len(results)}\n")

    if len(successful) > 0:
        print("Registered Models:")
        print("-" * 80)
        for _, row in successful.iterrows():
            print(
                f"  • {row['model_name']:<25} v{row['version']:<5} Accuracy: {row['accuracy']:.2%}"
            )
        print()

    if len(failed) > 0:
        print("Failed Registrations:")
        print("-" * 80)
        for _, row in failed.iterrows():
            print(f"  • {row['model_name']:<25} Error: {row.get('error', 'Unknown')}")
        print()

    # Save report
    report_dir = Path("reports/model_registry")
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = report_dir / f"registry_report_{timestamp}.csv"
    df.to_csv(report_file, index=False)

    print(f"📄 Report saved to: {report_file}")

    # Generate MLflow UI links
    print(f"\n{'='*80}")
    print("MLflow Model Registry Links")
    print(f"{'='*80}\n")
    print(f"Registry Home: {MLFLOW_TRACKING_URI}/#/models")

    for _, row in successful.iterrows():
        model_url = f"{MLFLOW_TRACKING_URI}/#/models/{row['model_name']}"
        print(f"  • {row['model_name']}: {model_url}")

    print()


def list_current_production_models():
    """List all models currently in Production stage"""
    client = MlflowClient()

    print(f"\n{'='*80}")
    print("CURRENT PRODUCTION MODELS")
    print(f"{'='*80}\n")

    try:
        models = client.search_registered_models()

        production_models = []
        for model in models:
            for version in model.latest_versions:
                if version.current_stage == "Production":
                    production_models.append(
                        {
                            "name": model.name,
                            "version": version.version,
                            "run_id": version.run_id,
                            "status": version.status,
                        }
                    )

        if production_models:
            for pm in production_models:
                print(
                    f"  • {pm['name']:<30} v{pm['version']:<5} (run: {pm['run_id'][:8]}...)"
                )
        else:
            print("  No models currently in Production stage")

        print()

    except Exception as e:
        print(f"  Error listing models: {e}\n")


def main():
    print("\n" + "=" * 80)
    print("MLflow Model Registration Script")
    print("=" * 80)
    print(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Models to Register: {len(TOP_MODELS)}")
    print("=" * 80 + "\n")

    # Check MLflow connection
    try:
        client = MlflowClient()
        experiments = client.search_experiments()
        print(f"✓ Connected to MLflow (found {len(experiments)} experiments)\n")
    except Exception as e:
        print(f"❌ Cannot connect to MLflow: {e}")
        print("\nPlease ensure MLflow server is running:")
        print("  mlflow server --backend-store-uri sqlite:///mlflow.db \\")
        print("                --default-artifact-root ./mlruns \\")
        print("                --host 0.0.0.0 --port 5000")
        return

    # Find the experiment with our trained models
    experiment_name = None
    for exp in experiments:
        if "Real_Datasets_Training" in exp.name:
            experiment_name = exp.name
            break

    if not experiment_name:
        print("❌ Cannot find training experiment")
        print("Available experiments:")
        for exp in experiments:
            print(f"  - {exp.name}")
        return

    print(f"Using experiment: {experiment_name}\n")

    # List current production models
    list_current_production_models()

    # Register each model
    results = []
    for model_info in TOP_MODELS:
        result = register_model(model_info, experiment_name)
        results.append(result)

    # Generate report
    generate_registry_report(results)

    print(f"\n{'='*80}")
    print("✅ Model Registration Complete!")
    print(f"{'='*80}\n")
    print("Next Steps:")
    print("  1. View models in MLflow UI: http://127.0.0.1:5000/#/models")
    print("  2. Test model loading: python scripts/test_model_loading.py")
    print("  3. Build production API: python api/production_api.py")
    print()


if __name__ == "__main__":
    main()
