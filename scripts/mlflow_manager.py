#!/usr/bin/env python3
"""
MLflow Experiment Management and Model Registry Setup

This script provides centralized MLflow configuration, experiment management,
and model registry operations for the healthcare recommendation platform.
"""

import os
import mlflow
import mlflow.sklearn
from pathlib import Path
from typing import Any, Dict, Optional, List
import logging
import yaml
from datetime import datetime

logger = logging.getLogger(__name__)


class MLflowManager:
    """Centralized MLflow management for experiments and model registry."""

    def __init__(
        self, tracking_uri: Optional[str] = None, experiment_name: Optional[str] = None
    ):
        """Initialize MLflow manager with configuration."""
        self.tracking_uri = tracking_uri or self._get_default_tracking_uri()
        self.experiment_name = experiment_name or "healthcare-ensemble-models"

        # Set up MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        self.experiment_id = self._setup_experiment()

        logger.info(f"MLflow tracking URI: {self.tracking_uri}")
        logger.info(f"Experiment: {self.experiment_name} (ID: {self.experiment_id})")

    def _get_default_tracking_uri(self) -> str:
        """Get default tracking URI from environment or use local file store."""
        # Check for environment variable first
        if "MLFLOW_TRACKING_URI" in os.environ:
            return os.environ["MLFLOW_TRACKING_URI"]

        # Use local file store
        mlruns_path = Path.cwd() / "mlruns"
        mlruns_path.mkdir(exist_ok=True)
        return f"file://{mlruns_path.absolute()}"

    def _setup_experiment(self) -> str:
        """Create or get existing experiment."""
        try:
            experiment_id = mlflow.create_experiment(
                name=self.experiment_name,
                tags={
                    "project": "hybrid-xai-healthcare",
                    "version": "0.1.0",
                    "created_at": datetime.now().isoformat(),
                    "description": "Explainable ensemble models for personalized treatment recommendation",
                },
            )
            logger.info(f"Created new experiment: {self.experiment_name}")
        except mlflow.exceptions.MlflowException:
            # Experiment already exists
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: {self.experiment_name}")

        mlflow.set_experiment(self.experiment_name)
        return experiment_id

    def start_run(
        self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None
    ) -> mlflow.ActiveRun:
        """Start a new MLflow run with standard tags."""
        default_tags = {
            "project": "hybrid-xai-healthcare",
            "environment": os.environ.get("ENVIRONMENT", "development"),
            "user": os.environ.get("USER", "unknown"),
            "timestamp": datetime.now().isoformat(),
        }

        if tags:
            default_tags.update(tags)

        return mlflow.start_run(run_name=run_name, tags=default_tags)

    def log_dataset_info(
        self, dataset_name: str, dataset_config: Dict[str, Any]
    ) -> None:
        """Log dataset information to current run."""
        mlflow.log_param("dataset_name", dataset_name)
        mlflow.log_param("dataset_path", dataset_config.get("raw_path", "unknown"))
        mlflow.log_param(
            "target_column", dataset_config.get("target_column", "unknown")
        )

        # Log dataset-specific overrides
        overrides = dataset_config.get("overrides", {})
        if overrides:
            for key, value in overrides.items():
                mlflow.log_param(f"override_{key}", str(value))

    def log_preprocessing_params(self, preprocessing_config: Dict[str, Any]) -> None:
        """Log preprocessing parameters."""
        if "tabular" in preprocessing_config:
            tabular = preprocessing_config["tabular"]
            mlflow.log_param(
                "numeric_imputation", tabular.get("numeric_imputation", "median")
            )
            mlflow.log_param(
                "categorical_imputation",
                tabular.get("categorical_imputation", "most_frequent"),
            )
            mlflow.log_param("scaling_method", tabular.get("scaling_method", "none"))

    def log_training_params(
        self,
        class_weight: Optional[str] = None,
        smote_applied: bool = False,
        imbalance_ratio: float = 1.0,
        **kwargs,
    ) -> None:
        """Log training-specific parameters."""
        mlflow.log_param("class_weight", class_weight or "none")
        mlflow.log_param("smote_applied", smote_applied)
        mlflow.log_param("imbalance_ratio", imbalance_ratio)

        # Log additional training parameters
        for key, value in kwargs.items():
            mlflow.log_param(key, value)

    def log_model_metrics(self, metrics: Dict[str, float], prefix: str = "") -> None:
        """Log model performance metrics."""
        for metric_name, value in metrics.items():
            metric_key = f"{prefix}_{metric_name}" if prefix else metric_name
            mlflow.log_metric(metric_key, float(value))

    def log_ensemble_model(
        self,
        model: Any,
        model_name: str,
        signature: Optional[mlflow.models.ModelSignature] = None,
        input_example: Optional[Any] = None,
        pip_requirements: Optional[List[str]] = None,
    ) -> str:
        """Log model to MLflow with proper artifacts."""

        # Default pip requirements for healthcare models
        default_requirements = [
            "scikit-learn>=1.0.0",
            "pandas>=1.3.0",
            "numpy>=1.21.0",
            "shap>=0.41.0",
            "lime>=0.2.0.1",
        ]

        if pip_requirements:
            requirements = list(set(default_requirements + pip_requirements))
        else:
            requirements = default_requirements

        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=model_name,
            signature=signature,
            input_example=input_example,
            pip_requirements=requirements,
        )

        logger.info(f"Model logged: {model_info.model_uri}")
        return model_info.model_uri

    def register_model(
        self,
        model_uri: str,
        model_name: str,
        stage: str = "Staging",
        description: Optional[str] = None,
    ) -> None:
        """Register model in MLflow Model Registry."""
        try:
            # Register the model
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                tags={
                    "project": "hybrid-xai-healthcare",
                    "registered_at": datetime.now().isoformat(),
                },
            )

            # Transition to specified stage
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name, version=model_version.version, stage=stage
            )

            # Add description if provided
            if description:
                client.update_model_version(
                    name=model_name,
                    version=model_version.version,
                    description=description,
                )

            logger.info(
                f"Model registered: {model_name} v{model_version.version} -> {stage}"
            )

        except mlflow.exceptions.MlflowException as e:
            logger.error(f"Failed to register model: {e}")
            raise

    def get_best_model(
        self, model_name: str, stage: str = "Production"
    ) -> Optional[Any]:
        """Retrieve the best model from registry."""
        try:
            client = mlflow.tracking.MlflowClient()
            model_version = client.get_latest_versions(name=model_name, stages=[stage])

            if not model_version:
                logger.warning(f"No model found in {stage} stage for {model_name}")
                return None

            model_uri = f"models:/{model_name}/{stage}"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Loaded model: {model_name} from {stage}")
            return model

        except mlflow.exceptions.MlflowException as e:
            logger.error(f"Failed to load model: {e}")
            return None

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments with their metadata."""
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()

        result = []
        for exp in experiments:
            result.append(
                {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "lifecycle_stage": exp.lifecycle_stage,
                    "artifact_location": exp.artifact_location,
                    "tags": exp.tags,
                }
            )

        return result

    def cleanup_old_runs(self, max_runs: int = 100) -> None:
        """Clean up old runs to prevent storage bloat."""
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(
            experiment_ids=[self.experiment_id],
            max_results=max_runs + 50,  # Get extra to identify old ones
        )

        if len(runs) > max_runs:
            runs_to_delete = runs[max_runs:]
            logger.info(f"Deleting {len(runs_to_delete)} old runs")

            for run in runs_to_delete:
                try:
                    client.delete_run(run.info.run_id)
                except mlflow.exceptions.MlflowException as e:
                    logger.warning(f"Failed to delete run {run.info.run_id}: {e}")


def setup_mlflow_server() -> None:
    """Set up MLflow tracking server with proper configuration."""

    # Create necessary directories
    mlruns_dir = Path("mlruns")
    mlruns_dir.mkdir(exist_ok=True)

    # Create MLflow server configuration
    config = {
        "backend_store_uri": "sqlite:///mlflow.db",
        "default_artifact_root": "./mlruns",
        "host": "0.0.0.0",
        "port": 5000,
    }

    # Save configuration
    config_path = Path("config/mlflow_config.yaml")
    config_path.parent.mkdir(exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"MLflow configuration saved to {config_path}")
    logger.info("To start MLflow server, run:")
    logger.info(
        f"mlflow server --backend-store-uri {config['backend_store_uri']} --default-artifact-root {config['default_artifact_root']} --host {config['host']} --port {config['port']}"
    )


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Setup MLflow server configuration
    setup_mlflow_server()

    # Test MLflow manager
    manager = MLflowManager()

    # List current experiments
    experiments = manager.list_experiments()
    print("\nCurrent experiments:")
    for exp in experiments:
        print(f"  - {exp['name']} (ID: {exp['experiment_id']})")

    print(f"\nMLflow setup complete! Tracking URI: {manager.tracking_uri}")
