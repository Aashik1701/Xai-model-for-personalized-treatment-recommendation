"""
Data loading and processing utilities for healthcare AI models.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Data loader for healthcare datasets.
    
    Handles loading, validation, and basic preprocessing of healthcare data.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize data loader.
        
        Args:
            config: Configuration dictionary for data loading
        """
        self.config = config or {}
        self.data = None
        self.metadata = {}
        
    def load_data(self, 
                  file_path: Union[str, Path],
                  file_format: str = "auto") -> pd.DataFrame:
        """
        Load data from file.
        
        Args:
            file_path: Path to data file
            file_format: Format of the file ("csv", "parquet", "json", "auto")
            
        Returns:
            Loaded DataFrame
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")
            
            if file_format == "auto":
                file_format = file_path.suffix.lower().lstrip('.')
            
            logger.info(f"Loading data from {file_path} (format: {file_format})")
            
            if file_format == "csv":
                self.data = pd.read_csv(file_path)
            elif file_format == "parquet":
                self.data = pd.read_parquet(file_path)
            elif file_format == "json":
                self.data = pd.read_json(file_path)
            elif file_format in ["xlsx", "xls"]:
                self.data = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            # Store metadata
            self.metadata = {
                "file_path": str(file_path),
                "file_format": file_format,
                "shape": self.data.shape,
                "columns": list(self.data.columns),
                "dtypes": dict(self.data.dtypes),
                "missing_values": dict(self.data.isnull().sum()),
                "memory_usage": self.data.memory_usage(deep=True).sum()
            }
            
            logger.info(f"Data loaded successfully: {self.data.shape}")
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def validate_data(self, 
                     required_columns: Optional[List[str]] = None,
                     target_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate loaded data.
        
        Args:
            required_columns: List of required column names
            target_column: Name of target column
            
        Returns:
            Validation report
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        validation_report = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "statistics": {}
        }
        
        try:
            # Check required columns
            if required_columns is not None:
                missing_columns = set(required_columns) - set(self.data.columns)
                if missing_columns:
                    validation_report["valid"] = False
                    validation_report["issues"].append(
                        f"Missing required columns: {missing_columns}"
                    )
            
            # Check target column
            if target_column and target_column not in self.data.columns:
                validation_report["valid"] = False
                validation_report["issues"].append(
                    f"Target column '{target_column}' not found"
                )
            
            # Check for empty dataset
            if self.data.empty:
                validation_report["valid"] = False
                validation_report["issues"].append("Dataset is empty")
            
            # Check for duplicate rows
            duplicates = self.data.duplicated().sum()
            if duplicates > 0:
                validation_report["warnings"].append(
                    f"Found {duplicates} duplicate rows"
                )
            
            # Check for missing values
            missing_counts = self.data.isnull().sum()
            high_missing = missing_counts[missing_counts > len(self.data) * 0.5]
            if not high_missing.empty:
                validation_report["warnings"].append(
                    f"Columns with >50% missing values: {list(high_missing.index)}"
                )
            
            # Generate statistics
            validation_report["statistics"] = {
                "total_rows": len(self.data),
                "total_columns": len(self.data.columns),
                "missing_values": dict(missing_counts),
                "duplicate_rows": duplicates,
                "numeric_columns": len(self.data.select_dtypes(include=[np.number]).columns),
                "categorical_columns": len(self.data.select_dtypes(include=['object']).columns)
            }
            
            logger.info(f"Data validation completed: {validation_report['valid']}")
            return validation_report
            
        except Exception as e:
            logger.error(f"Error during data validation: {e}")
            raise
    
    def split_data(self, 
                   target_column: str,
                   test_size: float = 0.2,
                   validation_size: float = 0.2,
                   random_state: int = 42,
                   stratify: bool = True) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            target_column: Name of target column
            test_size: Proportion of data for test set
            validation_size: Proportion of remaining data for validation set
            random_state: Random seed
            stratify: Whether to stratify split based on target
            
        Returns:
            Dictionary containing split datasets
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if target_column not in self.data.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        try:
            # Separate features and target
            X = self.data.drop(columns=[target_column])
            y = self.data[target_column]
            
            # Determine stratification
            stratify_param = y if stratify else None
            
            # First split: train+val vs test
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=random_state,
                stratify=stratify_param
            )
            
            # Second split: train vs validation
            if validation_size > 0:
                val_size = validation_size / (1 - test_size)
                stratify_temp = y_temp if stratify else None
                
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp,
                    test_size=val_size,
                    random_state=random_state,
                    stratify=stratify_temp
                )
                
                split_data = {
                    "X_train": X_train,
                    "X_val": X_val,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_val": y_val,
                    "y_test": y_test
                }
            else:
                split_data = {
                    "X_train": X_temp,
                    "X_test": X_test,
                    "y_train": y_temp,
                    "y_test": y_test
                }
            
            # Log split information
            logger.info("Data split completed:")
            for key, value in split_data.items():
                logger.info(f"  {key}: {value.shape}")
            
            return split_data
            
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            raise
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive data summary.
        
        Returns:
            Dictionary containing data summary statistics
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        try:
            # Provide backward-compatible top-level keys (shape, columns, missing_values)
            summary = {
                "shape": self.data.shape,
                "columns": list(self.data.columns),
                "basic_info": {
                    "shape": self.data.shape,
                    "memory_usage_mb": self.data.memory_usage(deep=True).sum() / 1024**2,
                    "columns": list(self.data.columns)
                },
                "data_types": dict(self.data.dtypes),
                "missing_values": dict(self.data.isnull().sum()),  # also exposed at top-level
                "statistics": {}
            }
            
            # Numeric columns statistics
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                summary["statistics"]["numeric"] = self.data[numeric_cols].describe().to_dict()
            
            # Categorical columns statistics
            categorical_cols = self.data.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                summary["statistics"]["categorical"] = {}
                for col in categorical_cols:
                    summary["statistics"]["categorical"][col] = {
                        "unique_values": self.data[col].nunique(),
                        "most_frequent": self.data[col].mode().iloc[0] if len(self.data[col].mode()) > 0 else None,
                        "frequency": dict(self.data[col].value_counts().head())
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating data summary: {e}")
            raise
    
    def save_processed_data(self, 
                           data: pd.DataFrame,
                           file_path: Union[str, Path],
                           file_format: str = "parquet") -> None:
        """
        Save processed data to file.
        
        Args:
            data: DataFrame to save
            file_path: Output file path
            file_format: Output file format
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if file_format == "csv":
                data.to_csv(file_path, index=False)
            elif file_format == "parquet":
                data.to_parquet(file_path, index=False)
            elif file_format == "json":
                data.to_json(file_path, orient="records")
            else:
                raise ValueError(f"Unsupported output format: {file_format}")
            
            logger.info(f"Data saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise
