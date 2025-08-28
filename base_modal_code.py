# Create key implementation files for the hybrid explainable AI system

# 1. Base Model Class
base_model_code = '''"""
Base model abstract class for hybrid explainable AI healthcare models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import logging

logger = logging.getLogger(__name__)


class BaseHybridModel(ABC, BaseEstimator):
    """
    Abstract base class for hybrid explainable AI models in healthcare.
    
    This class defines the interface that all hybrid models must implement,
    including training, prediction, and explainability methods.
    """
    
    def __init__(self, 
                 model_name: str,
                 config: Dict[str, Any],
                 random_state: Optional[int] = 42):
        """
        Initialize the base hybrid model.
        
        Args:
            model_name: Name of the model
            config: Configuration dictionary
            random_state: Random seed for reproducibility
        """
        self.model_name = model_name
        self.config = config
        self.random_state = random_state
        self.is_fitted = False
        self.feature_names = None
        self.target_names = None
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, 
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            **kwargs) -> "BaseHybridModel":
        """
        Train the hybrid model on the provided data.
        
        Args:
            X: Training features
            y: Training targets
            validation_data: Optional validation data tuple (X_val, y_val)
            **kwargs: Additional training parameters
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
            
        Returns:
            Predictions array
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Probability predictions array
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get global feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass
    
    def set_feature_names(self, feature_names: List[str]) -> None:
        """Set feature names for interpretability."""
        self.feature_names = feature_names
        
    def set_target_names(self, target_names: List[str]) -> None:
        """Set target names for interpretability."""
        self.target_names = target_names
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata and configuration."""
        return {
            "model_name": self.model_name,
            "config": self.config,
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names,
            "target_names": self.target_names
        }
'''

# 2. SHAP Explainer Implementation
shap_explainer_code = '''"""
SHAP-based explainer for hybrid healthcare models.
"""

import shap
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from ..models.base_model import BaseHybridModel
import logging

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP explainer for generating model interpretations.
    
    This class provides comprehensive SHAP-based explanations for
    hybrid healthcare AI models.
    """
    
    def __init__(self, 
                 model: BaseHybridModel,
                 explainer_type: str = "auto",
                 background_data: Optional[np.ndarray] = None):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained hybrid model
            explainer_type: Type of SHAP explainer ("tree", "kernel", "deep", "auto")
            background_data: Background dataset for kernel explainer
        """
        self.model = model
        self.explainer_type = explainer_type
        self.background_data = background_data
        self.explainer = None
        self._initialize_explainer()
        
    def _initialize_explainer(self) -> None:
        """Initialize the appropriate SHAP explainer based on model type."""
        try:
            if self.explainer_type == "auto":
                # Auto-detect explainer type based on model
                if hasattr(self.model, 'tree_'):
                    self.explainer = shap.TreeExplainer(self.model)
                    logger.info("Using TreeExplainer")
                else:
                    self.explainer = shap.KernelExplainer(
                        self.model.predict_proba, 
                        self.background_data
                    )
                    logger.info("Using KernelExplainer")
            elif self.explainer_type == "tree":
                self.explainer = shap.TreeExplainer(self.model)
            elif self.explainer_type == "kernel":
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba, 
                    self.background_data
                )
            elif self.explainer_type == "deep":
                self.explainer = shap.DeepExplainer(
                    self.model, 
                    self.background_data
                )
            else:
                raise ValueError(f"Unknown explainer type: {self.explainer_type}")
                
        except Exception as e:
            logger.error(f"Error initializing SHAP explainer: {e}")
            raise
    
    def explain_instance(self, 
                        instance: np.ndarray,
                        return_dict: bool = True) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Generate SHAP explanations for a single instance.
        
        Args:
            instance: Single instance to explain
            return_dict: If True, return structured explanation dictionary
            
        Returns:
            SHAP values or explanation dictionary
        """
        try:
            if instance.ndim == 1:
                instance = instance.reshape(1, -1)
                
            shap_values = self.explainer.shap_values(instance)
            
            if return_dict:
                explanation = {
                    "shap_values": shap_values,
                    "base_value": self.explainer.expected_value,
                    "data": instance,
                    "feature_names": self.model.feature_names,
                    "prediction": self.model.predict(instance)[0],
                    "prediction_proba": self.model.predict_proba(instance)[0]
                }
                
                # Add feature contributions
                if self.model.feature_names:
                    if isinstance(shap_values, list):  # Multi-class
                        explanation["feature_contributions"] = {
                            f"class_{i}": dict(zip(self.model.feature_names, shap_vals[0]))
                            for i, shap_vals in enumerate(shap_values)
                        }
                    else:  # Binary or regression
                        explanation["feature_contributions"] = dict(
                            zip(self.model.feature_names, shap_values[0])
                        )
                
                return explanation
            else:
                return shap_values
                
        except Exception as e:
            logger.error(f"Error explaining instance: {e}")
            raise
    
    def explain_batch(self, 
                     X: np.ndarray,
                     batch_size: int = 100) -> Dict[str, Any]:
        """
        Generate SHAP explanations for a batch of instances.
        
        Args:
            X: Batch of instances to explain
            batch_size: Size of processing batches
            
        Returns:
            Dictionary containing batch explanations
        """
        try:
            n_samples = X.shape[0]
            all_shap_values = []
            
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch = X[start_idx:end_idx]
                
                batch_shap_values = self.explainer.shap_values(batch)
                all_shap_values.append(batch_shap_values)
            
            # Concatenate all batches
            if isinstance(all_shap_values[0], list):  # Multi-class
                shap_values = [
                    np.vstack([batch[i] for batch in all_shap_values])
                    for i in range(len(all_shap_values[0]))
                ]
            else:  # Binary or regression
                shap_values = np.vstack(all_shap_values)
            
            return {
                "shap_values": shap_values,
                "base_value": self.explainer.expected_value,
                "data": X,
                "feature_names": self.model.feature_names,
                "predictions": self.model.predict(X),
                "predictions_proba": self.model.predict_proba(X)
            }
            
        except Exception as e:
            logger.error(f"Error explaining batch: {e}")
            raise
    
    def plot_waterfall(self, 
                      explanation: Dict[str, Any],
                      class_idx: int = 0,
                      save_path: Optional[str] = None) -> None:
        """
        Create a waterfall plot for SHAP explanations.
        
        Args:
            explanation: Explanation dictionary from explain_instance
            class_idx: Class index for multi-class problems
            save_path: Path to save the plot
        """
        try:
            shap_values = explanation["shap_values"]
            if isinstance(shap_values, list):
                shap_values = shap_values[class_idx]
            
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=explanation["base_value"],
                    data=explanation["data"][0],
                    feature_names=explanation["feature_names"]
                )
            )
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close()
                
        except Exception as e:
            logger.error(f"Error creating waterfall plot: {e}")
            raise
    
    def plot_summary(self, 
                    explanations: Dict[str, Any],
                    plot_type: str = "dot",
                    save_path: Optional[str] = None) -> None:
        """
        Create a summary plot for batch explanations.
        
        Args:
            explanations: Batch explanations from explain_batch
            plot_type: Type of summary plot ("dot", "bar")
            save_path: Path to save the plot
        """
        try:
            shap.summary_plot(
                explanations["shap_values"],
                explanations["data"],
                feature_names=explanations["feature_names"],
                plot_type=plot_type,
                show=False
            )
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close()
                
        except Exception as e:
            logger.error(f"Error creating summary plot: {e}")
            raise
    
    def get_global_importance(self, 
                            explanations: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate global feature importance from batch explanations.
        
        Args:
            explanations: Batch explanations from explain_batch
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        try:
            shap_values = explanations["shap_values"]
            if isinstance(shap_values, list):
                # For multi-class, average across classes
                importance_scores = np.mean([
                    np.mean(np.abs(class_shap), axis=0)
                    for class_shap in shap_values
                ], axis=0)
            else:
                importance_scores = np.mean(np.abs(shap_values), axis=0)
            
            if self.model.feature_names:
                return dict(zip(self.model.feature_names, importance_scores))
            else:
                return {f"feature_{i}": score 
                       for i, score in enumerate(importance_scores)}
                
        except Exception as e:
            logger.error(f"Error calculating global importance: {e}")
            raise
'''

# 3. Configuration Management
config_code = '''"""
Configuration management for hybrid explainable AI healthcare models.
"""

import yaml
import os
from typing import Any, Dict, Optional
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
'''

print("🔧 Core Implementation Files Created:")
print("=" * 45)
print("✅ base_model.py - Abstract base model class")
print("✅ shap_explainer.py - SHAP explanation system")
print("✅ config.py - Configuration management")
print("\n🎯 Key Features Implemented:")
print("- Abstract base class for all hybrid models")
print("- Comprehensive SHAP explainer with multiple plot types") 
print("- Centralized configuration management system")
print("- Type hints and proper error handling")
print("- Logging integration for debugging")
print("- Support for multi-class and regression tasks")