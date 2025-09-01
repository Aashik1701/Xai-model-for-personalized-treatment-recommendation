"""
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
    
    def save_model(self, file_path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            file_path: Path to save the model
        """
        import joblib
        if not self.is_fitted:
            logger.warning("Model is not fitted. Saving untrained model.")
        
        joblib.dump(self, file_path)
        logger.info(f"Model saved to {file_path}")
    
    @classmethod
    def load_model(cls, file_path: str) -> "BaseHybridModel":
        """
        Load a trained model from disk.
        
        Args:
            file_path: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        import joblib
        model = joblib.load(file_path)
        logger.info(f"Model loaded from {file_path}")
        return model
    
    def validate_input(self, X: np.ndarray) -> np.ndarray:
        """
        Validate and prepare input data.
        
        Args:
            X: Input features
            
        Returns:
            Validated input array
        """
        if not isinstance(X, np.ndarray):
            if isinstance(X, pd.DataFrame):
                X = X.values
            else:
                X = np.array(X)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        return X
    
    def get_training_history(self) -> Dict[str, Any]:
        """
        Get training history and metrics.
        
        Returns:
            Dictionary containing training information
        """
        if hasattr(self, 'training_history_'):
            return self.training_history_
        else:
            logger.warning("No training history available")
            return {}
