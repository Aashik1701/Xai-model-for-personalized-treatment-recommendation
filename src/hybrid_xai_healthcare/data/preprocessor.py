"""
Data preprocessing utilities for healthcare AI models.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging

logger = logging.getLogger(__name__)


class HealthcareDataPreprocessor:
    """
    Comprehensive data preprocessor for healthcare datasets.
    
    Handles missing values, encoding, scaling, and feature engineering
    specific to healthcare data.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize preprocessor.
        
        Args:
            config: Configuration dictionary for preprocessing steps
        """
        self.config = config or {}
        self.preprocessor = None
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        self.is_fitted = False
        
    def fit(self, 
            X: pd.DataFrame, 
            y: Optional[pd.Series] = None) -> "HealthcareDataPreprocessor":
        """
        Fit the preprocessor on training data.
        
        Args:
            X: Training features
            y: Training targets (optional)
            
        Returns:
            Self for method chaining
        """
        try:
            logger.info("Fitting preprocessor...")
            
            # Store feature names
            self.feature_names_in_ = list(X.columns)
            
            # Identify column types
            numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Create preprocessing pipelines
            numeric_pipeline = self._create_numeric_pipeline()
            categorical_pipeline = self._create_categorical_pipeline()
            
            # Combine pipelines
            transformers = []
            if numeric_features:
                transformers.append(('num', numeric_pipeline, numeric_features))
            if categorical_features:
                transformers.append(('cat', categorical_pipeline, categorical_features))
            
            if not transformers:
                raise ValueError("No features to preprocess")
            
            self.preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder='passthrough'
            )
            
            # Fit the preprocessor
            self.preprocessor.fit(X)
            self.is_fitted = True
            
            # Generate output feature names
            self._generate_output_feature_names(X)
            
            logger.info(f"Preprocessor fitted successfully. Input features: {len(self.feature_names_in_)}, "
                       f"Output features: {len(self.feature_names_out_)}")
            
            return self
            
        except Exception as e:
            logger.error(f"Error fitting preprocessor: {e}")
            raise
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted preprocessor.
        
        Args:
            X: Features to transform
            
        Returns:
            Transformed feature array
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        try:
            logger.info(f"Transforming data with shape {X.shape}")
            
            # Check if input features match
            if list(X.columns) != self.feature_names_in_:
                logger.warning("Input feature names don't match training features")
            
            transformed = self.preprocessor.transform(X)
            
            logger.info(f"Data transformed to shape {transformed.shape}")
            return transformed
            
        except Exception as e:
            logger.error(f"Error transforming data: {e}")
            raise
    
    def fit_transform(self, 
                     X: pd.DataFrame, 
                     y: Optional[pd.Series] = None) -> np.ndarray:
        """
        Fit preprocessor and transform data in one step.
        
        Args:
            X: Training features
            y: Training targets (optional)
            
        Returns:
            Transformed feature array
        """
        return self.fit(X, y).transform(X)
    
    def _create_numeric_pipeline(self) -> Pipeline:
        """Create preprocessing pipeline for numeric features."""
        steps = []
        
        # Imputation
        imputation_strategy = self.config.get('numeric_imputation', 'median')
        if imputation_strategy == 'knn':
            imputer = KNNImputer(n_neighbors=5)
        else:
            imputer = SimpleImputer(strategy=imputation_strategy)
        steps.append(('imputer', imputer))
        
        # Scaling
        scaling_method = self.config.get('scaling_method', 'standard')
        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
        elif scaling_method == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        steps.append(('scaler', scaler))
        
        return Pipeline(steps)
    
    def _create_categorical_pipeline(self) -> Pipeline:
        """Create preprocessing pipeline for categorical features."""
        steps = []
        
        # Imputation
        imputation_strategy = self.config.get('categorical_imputation', 'most_frequent')
        imputer = SimpleImputer(strategy=imputation_strategy)
        steps.append(('imputer', imputer))
        
        # Encoding
        encoding_method = self.config.get('encoding_method', 'onehot')
        if encoding_method == 'onehot':
            # Handle sklearn version differences: sparse_output introduced in 1.2
            try:  # Prefer new API
                encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            except TypeError:  # Fallback for older sklearn (<1.2)
                encoder = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
        elif encoding_method == 'ordinal':
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        else:
            encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        steps.append(('encoder', encoder))
        
        return Pipeline(steps)
    
    def _generate_output_feature_names(self, X: pd.DataFrame) -> None:
        """Generate names for output features after transformation."""
        try:
            feature_names = []
            
            # Get feature names from each transformer
            for name, transformer, features in self.preprocessor.transformers_:
                if name == 'num':
                    # Numeric features keep their original names
                    feature_names.extend(features)
                elif name == 'cat':
                    # Categorical features may be expanded due to encoding
                    if hasattr(transformer.named_steps['encoder'], 'get_feature_names_out'):
                        encoded_names = transformer.named_steps['encoder'].get_feature_names_out(features)
                        feature_names.extend(encoded_names)
                    else:
                        feature_names.extend(features)
                else:
                    # Passthrough features
                    feature_names.extend(features)
            
            self.feature_names_out_ = feature_names
            
        except Exception as e:
            logger.warning(f"Could not generate output feature names: {e}")
            self.feature_names_out_ = [f"feature_{i}" for i in range(len(self.feature_names_in_))]
    
    def get_feature_names_out(self) -> List[str]:
        """Get output feature names after transformation."""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        return self.feature_names_out_
    
    def inverse_transform(self, X_transformed: np.ndarray) -> pd.DataFrame:
        """
        Inverse transform data back to original space (if possible).
        
        Args:
            X_transformed: Transformed feature array
            
        Returns:
            DataFrame with original feature values
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        try:
            # Note: Inverse transform may not be perfect due to information loss
            # in categorical encoding and imputation
            inverse_transformed = self.preprocessor.inverse_transform(X_transformed)
            
            return pd.DataFrame(
                inverse_transformed,
                columns=self.feature_names_in_
            )
            
        except Exception as e:
            logger.error(f"Error in inverse transform: {e}")
            raise
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """
        Get information about the preprocessing steps and transformations.
        
        Returns:
            Dictionary containing preprocessing information
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        info = {
            "input_features": len(self.feature_names_in_),
            "output_features": len(self.feature_names_out_),
            "feature_names_in": self.feature_names_in_,
            "feature_names_out": self.feature_names_out_,
            "config": self.config,
            "transformers": []
        }
        
        # Get information about each transformer
        for name, transformer, features in self.preprocessor.transformers_:
            transformer_info = {
                "name": name,
                "type": type(transformer).__name__,
                "features": list(features) if hasattr(features, '__iter__') else features,
                "steps": []
            }
            
            # Get steps for pipeline transformers
            if hasattr(transformer, 'steps'):
                for step_name, step_transformer in transformer.steps:
                    transformer_info["steps"].append({
                        "name": step_name,
                        "type": type(step_transformer).__name__
                    })
            
            info["transformers"].append(transformer_info)
        
        return info
    
    def save_preprocessor(self, file_path: str) -> None:
        """
        Save fitted preprocessor to disk.
        
        Args:
            file_path: Path to save the preprocessor
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        import joblib
        joblib.dump(self, file_path)
        logger.info(f"Preprocessor saved to {file_path}")
    
    @classmethod
    def load_preprocessor(cls, file_path: str) -> "HealthcareDataPreprocessor":
        """
        Load fitted preprocessor from disk.
        
        Args:
            file_path: Path to saved preprocessor
            
        Returns:
            Loaded preprocessor instance
        """
        import joblib
        preprocessor = joblib.load(file_path)
        logger.info(f"Preprocessor loaded from {file_path}")
        return preprocessor
