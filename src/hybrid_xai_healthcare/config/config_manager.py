"""
Configuration management for hybrid explainable AI healthcare models.
"""

import yaml
import os
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model training and inference."""
    model_type: str
    hyperparameters: Dict[str, Any]
    ensemble_config: Optional[Dict[str, Any]] = None
    explainability_config: Optional[Dict[str, Any]] = None


@dataclass
class DataConfig:
    """Configuration for data processing and loading."""
    data_path: str
    features: List[str]
    target_column: str
    preprocessing_steps: List[str]
    validation_split: float = 0.2
    test_split: float = 0.1


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    epochs: int
    batch_size: int
    learning_rate: float
    early_stopping: bool = True
    patience: int = 10
    cross_validation_folds: int = 5


@dataclass
class ExplainabilityConfig:
    """Configuration for explainability methods."""
    methods: List[str]  # ["shap", "lime", "attention"]
    shap_config: Dict[str, Any]
    lime_config: Dict[str, Any]
    attention_config: Dict[str, Any]


class ConfigManager:
    """
    Centralized configuration management for the healthcare AI system.
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.configs = {}
        self._load_all_configs()
    
    def _load_all_configs(self) -> None:
        """Load all configuration files from the config directory."""
        try:
            config_files = {
                "model": "model_config.yaml",
                "data": "data_config.yaml", 
                "training": "training_config.yaml",
                "explainability": "explainability_config.yaml"
            }
            
            for config_name, filename in config_files.items():
                config_path = self.config_dir / filename
                if config_path.exists():
                    self.configs[config_name] = self._load_yaml(config_path)
                    logger.info(f"Loaded {config_name} configuration")
                else:
                    logger.warning(f"Configuration file not found: {config_path}")
                    
        except Exception as e:
            logger.error(f"Error loading configurations: {e}")
            raise
    
    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(file_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Error loading YAML file {file_path}: {e}")
            raise
    
    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get model configuration by name."""
        try:
            model_configs = self.configs.get("model", {})
            if model_name not in model_configs:
                raise ValueError(f"Model configuration not found: {model_name}")
            
            config = model_configs[model_name]
            return ModelConfig(
                model_type=config["model_type"],
                hyperparameters=config["hyperparameters"],
                ensemble_config=config.get("ensemble_config"),
                explainability_config=config.get("explainability_config")
            )
        except Exception as e:
            logger.error(f"Error getting model config: {e}")
            raise
    
    def get_data_config(self) -> DataConfig:
        """Get data configuration."""
        try:
            config = self.configs.get("data", {})
            return DataConfig(
                data_path=config["data_path"],
                features=config["features"],
                target_column=config["target_column"],
                preprocessing_steps=config["preprocessing_steps"],
                validation_split=config.get("validation_split", 0.2),
                test_split=config.get("test_split", 0.1)
            )
        except Exception as e:
            logger.error(f"Error getting data config: {e}")
            raise
    
    def get_training_config(self) -> TrainingConfig:
        """Get training configuration."""
        try:
            config = self.configs.get("training", {})
            return TrainingConfig(
                epochs=config["epochs"],
                batch_size=config["batch_size"],
                learning_rate=config["learning_rate"],
                early_stopping=config.get("early_stopping", True),
                patience=config.get("patience", 10),
                cross_validation_folds=config.get("cross_validation_folds", 5)
            )
        except Exception as e:
            logger.error(f"Error getting training config: {e}")
            raise
    
    def get_explainability_config(self) -> ExplainabilityConfig:
        """Get explainability configuration."""
        try:
            config = self.configs.get("explainability", {})
            return ExplainabilityConfig(
                methods=config["methods"],
                shap_config=config["shap_config"],
                lime_config=config["lime_config"],
                attention_config=config["attention_config"]
            )
        except Exception as e:
            logger.error(f"Error getting explainability config: {e}")
            raise
    
    def update_config(self, 
                     config_type: str, 
                     updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        try:
            if config_type not in self.configs:
                self.configs[config_type] = {}
            
            self.configs[config_type].update(updates)
            logger.info(f"Updated {config_type} configuration")
            
        except Exception as e:
            logger.error(f"Error updating config: {e}")
            raise
    
    def save_config(self, config_type: str) -> None:
        """Save configuration to file."""
        try:
            config_files = {
                "model": "model_config.yaml",
                "data": "data_config.yaml",
                "training": "training_config.yaml", 
                "explainability": "explainability_config.yaml"
            }
            
            if config_type not in config_files:
                raise ValueError(f"Unknown config type: {config_type}")
            
            file_path = self.config_dir / config_files[config_type]
            with open(file_path, 'w') as file:
                yaml.dump(self.configs[config_type], file, default_flow_style=False)
            
            logger.info(f"Saved {config_type} configuration to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            raise
    
    def get_all_configs(self) -> Dict[str, Any]:
        """Get all loaded configurations."""
        return self.configs.copy()
    
    def validate_configs(self) -> Dict[str, Any]:
        """
        Validate all loaded configurations.
        
        Returns:
            Validation report
        """
        validation_report = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Validate model config
            if "model" in self.configs:
                for model_name, model_config in self.configs["model"].items():
                    if "model_type" not in model_config:
                        validation_report["errors"].append(
                            f"Missing 'model_type' in {model_name} configuration"
                        )
                    if "hyperparameters" not in model_config:
                        validation_report["errors"].append(
                            f"Missing 'hyperparameters' in {model_name} configuration"
                        )
            
            # Validate data config
            if "data" in self.configs:
                data_config = self.configs["data"]
                required_fields = ["data_path", "features", "target_column"]
                for field in required_fields:
                    if field not in data_config:
                        validation_report["errors"].append(
                            f"Missing '{field}' in data configuration"
                        )
            
            # Validate training config
            if "training" in self.configs:
                training_config = self.configs["training"]
                required_fields = ["epochs", "batch_size", "learning_rate"]
                for field in required_fields:
                    if field not in training_config:
                        validation_report["errors"].append(
                            f"Missing '{field}' in training configuration"
                        )
            
            # Set validation status
            validation_report["valid"] = len(validation_report["errors"]) == 0
            
            return validation_report
            
        except Exception as e:
            logger.error(f"Error validating configs: {e}")
            validation_report["valid"] = False
            validation_report["errors"].append(f"Validation error: {e}")
            return validation_report
