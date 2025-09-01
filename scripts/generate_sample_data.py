"""
Generate synthetic healthcare data for testing the hybrid explainable AI system.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_healthcare_data(n_samples: int = 1000, 
                           random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic healthcare data for testing.
    
    Args:
        n_samples: Number of patient records to generate
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic patient data
    """
    np.random.seed(random_state)
    
    logger.info(f"Generating {n_samples} synthetic patient records...")
    
    # Generate base classification data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=2,
        n_classes=3,  # 3 treatment outcomes
        random_state=random_state
    )
    
    # Create realistic healthcare feature names
    feature_names = [
        'age', 'gender', 'bmi', 'blood_pressure_systolic', 'blood_pressure_diastolic',
        'glucose_level', 'cholesterol_total', 'cholesterol_hdl', 'cholesterol_ldl',
        'heart_rate', 'temperature', 'white_blood_cells', 'red_blood_cells', 
        'hemoglobin', 'platelets', 'creatinine', 'albumin', 'bilirubin',
        'respiratory_rate', 'oxygen_saturation'
    ]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    
    # Post-process features to make them more realistic
    df = _make_features_realistic(df)
    
    # Add categorical features
    df = _add_categorical_features(df, n_samples, random_state)
    
    # Add target variable
    target_mapping = {0: 'improved', 1: 'stable', 2: 'deteriorated'}
    df['treatment_outcome'] = [target_mapping[val] for val in y]
    
    # Add patient ID
    df['patient_id'] = range(1, n_samples + 1)
    
    # Reorder columns
    cols = ['patient_id'] + [col for col in df.columns if col != 'patient_id']
    df = df[cols]
    
    logger.info("Synthetic healthcare data generated successfully")
    return df


def _make_features_realistic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform features to have realistic healthcare value ranges.
    
    Args:
        df: DataFrame with normalized features
        
    Returns:
        DataFrame with realistic feature ranges
    """
    # Age: 18-90 years
    df['age'] = np.clip(df['age'] * 15 + 55, 18, 90)
    
    # Gender: binary (0=female, 1=male)
    df['gender'] = (df['gender'] > 0).astype(int)
    
    # BMI: 15-50
    df['bmi'] = np.clip(df['bmi'] * 8 + 25, 15, 50)
    
    # Blood pressure: systolic 90-200, diastolic 60-120
    df['blood_pressure_systolic'] = np.clip(df['blood_pressure_systolic'] * 30 + 120, 90, 200)
    df['blood_pressure_diastolic'] = np.clip(df['blood_pressure_diastolic'] * 20 + 80, 60, 120)
    
    # Glucose: 70-300 mg/dL
    df['glucose_level'] = np.clip(df['glucose_level'] * 50 + 100, 70, 300)
    
    # Cholesterol levels
    df['cholesterol_total'] = np.clip(df['cholesterol_total'] * 80 + 180, 120, 400)
    df['cholesterol_hdl'] = np.clip(df['cholesterol_hdl'] * 30 + 50, 20, 100)
    df['cholesterol_ldl'] = np.clip(df['cholesterol_ldl'] * 80 + 100, 50, 300)
    
    # Heart rate: 50-150 bpm
    df['heart_rate'] = np.clip(df['heart_rate'] * 30 + 70, 50, 150)
    
    # Temperature: 96-104°F
    df['temperature'] = np.clip(df['temperature'] * 2 + 98.6, 96, 104)
    
    # Blood counts
    df['white_blood_cells'] = np.clip(df['white_blood_cells'] * 5 + 7, 3, 15)  # thousands/μL
    df['red_blood_cells'] = np.clip(df['red_blood_cells'] * 2 + 4.5, 3.5, 6)  # millions/μL
    df['hemoglobin'] = np.clip(df['hemoglobin'] * 4 + 13, 8, 18)  # g/dL
    df['platelets'] = np.clip(df['platelets'] * 200 + 250, 100, 500)  # thousands/μL
    
    # Chemistry panel
    df['creatinine'] = np.clip(df['creatinine'] * 1 + 1.0, 0.5, 3.0)  # mg/dL
    df['albumin'] = np.clip(df['albumin'] * 1.5 + 4.0, 2.5, 5.5)  # g/dL
    df['bilirubin'] = np.clip(df['bilirubin'] * 2 + 1.0, 0.2, 5.0)  # mg/dL
    
    # Vital signs
    df['respiratory_rate'] = np.clip(df['respiratory_rate'] * 8 + 16, 10, 30)  # breaths/min
    df['oxygen_saturation'] = np.clip(df['oxygen_saturation'] * 5 + 97, 85, 100)  # %
    
    # Round appropriate features
    integer_features = ['age', 'gender', 'heart_rate', 'respiratory_rate']
    for feature in integer_features:
        df[feature] = df[feature].round().astype(int)
    
    # Round to 1 decimal place for other features
    float_features = [col for col in df.columns if col not in integer_features]
    for feature in float_features:
        df[feature] = df[feature].round(1)
    
    return df


def _add_categorical_features(df: pd.DataFrame, 
                            n_samples: int, 
                            random_state: int) -> pd.DataFrame:
    """
    Add categorical features to the dataset.
    
    Args:
        df: DataFrame with numeric features
        n_samples: Number of samples
        random_state: Random seed
        
    Returns:
        DataFrame with added categorical features
    """
    np.random.seed(random_state)
    
    # Symptoms (multi-hot encoding possibility)
    symptoms = ['fever', 'cough', 'fatigue', 'shortness_of_breath', 'chest_pain']
    for symptom in symptoms:
        df[f'symptom_{symptom}'] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    
    # Prior medical conditions
    conditions = ['diabetes', 'hypertension', 'heart_disease', 'kidney_disease']
    for condition in conditions:
        df[f'prior_condition_{condition}'] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    
    # Medication history
    df['medication_count'] = np.random.poisson(2, n_samples)
    
    # Family history
    df['family_history_score'] = np.random.choice([0, 1, 2, 3], n_samples, p=[0.4, 0.3, 0.2, 0.1])
    
    # Lifestyle factors
    df['smoking_status'] = np.random.choice(
        ['never', 'former', 'current'], 
        n_samples, 
        p=[0.6, 0.25, 0.15]
    )
    
    df['alcohol_consumption'] = np.random.choice(
        ['none', 'moderate', 'heavy'], 
        n_samples, 
        p=[0.4, 0.5, 0.1]
    )
    
    df['exercise_frequency'] = np.random.choice(
        ['never', 'rarely', 'weekly', 'daily'], 
        n_samples, 
        p=[0.2, 0.3, 0.3, 0.2]
    )
    
    # Hospital admission details
    df['admission_type'] = np.random.choice(
        ['emergency', 'urgent', 'elective'], 
        n_samples, 
        p=[0.3, 0.4, 0.3]
    )
    
    df['length_of_stay'] = np.random.gamma(2, 2, n_samples).round().astype(int)
    df['icu_admission'] = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    
    return df


def add_missing_values(df: pd.DataFrame, 
                      missing_rate: float = 0.05) -> pd.DataFrame:
    """
    Add missing values to simulate real-world healthcare data.
    
    Args:
        df: Complete DataFrame
        missing_rate: Proportion of values to make missing
        
    Returns:
        DataFrame with missing values
    """
    logger.info(f"Adding {missing_rate*100}% missing values...")
    
    df_with_missing = df.copy()
    
    # Select columns that can have missing values (exclude patient_id and treatment_outcome)
    exclude_cols = ['patient_id', 'treatment_outcome']
    cols_with_missing = [col for col in df.columns if col not in exclude_cols]
    
    # Add missing values randomly
    for col in cols_with_missing:
        n_missing = int(len(df) * missing_rate)
        missing_indices = np.random.choice(len(df), n_missing, replace=False)
        df_with_missing.loc[missing_indices, col] = np.nan
    
    return df_with_missing


def save_datasets(df: pd.DataFrame, output_dir: str = "data") -> None:
    """
    Save datasets in different formats and splits.
    
    Args:
        df: DataFrame to save
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    
    # Create directories
    (output_path / "raw").mkdir(parents=True, exist_ok=True)
    (output_path / "processed").mkdir(parents=True, exist_ok=True)
    
    # Save complete dataset
    df.to_csv(output_path / "raw" / "synthetic_healthcare_data.csv", index=False)
    logger.info(f"Saved complete dataset to {output_path / 'raw' / 'synthetic_healthcare_data.csv'}")
    
    # Save dataset with missing values
    df_missing = add_missing_values(df, missing_rate=0.05)
    df_missing.to_csv(output_path / "raw" / "synthetic_healthcare_data_with_missing.csv", index=False)
    logger.info(f"Saved dataset with missing values to {output_path / 'raw' / 'synthetic_healthcare_data_with_missing.csv'}")
    
    # Save train/test splits
    from sklearn.model_selection import train_test_split
    
    # Separate features and target
    X = df.drop(['patient_id', 'treatment_outcome'], axis=1)
    y = df['treatment_outcome']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save splits
    train_data = X_train.copy()
    train_data['treatment_outcome'] = y_train
    train_data.to_csv(output_path / "processed" / "train_data.csv", index=False)
    
    test_data = X_test.copy()
    test_data['treatment_outcome'] = y_test
    test_data.to_csv(output_path / "processed" / "test_data.csv", index=False)
    
    logger.info(f"Saved train/test splits to {output_path / 'processed'}")


def main():
    """Main function to generate and save synthetic healthcare data."""
    try:
        # Generate data
        df = generate_healthcare_data(n_samples=2000, random_state=42)
        
        # Display basic statistics
        logger.info(f"Generated dataset shape: {df.shape}")
        logger.info(f"Features: {list(df.columns)}")
        logger.info(f"Target distribution:\n{df['treatment_outcome'].value_counts()}")
        
        # Save datasets
        save_datasets(df)
        
        logger.info("Data generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error generating data: {e}")
        raise


if __name__ == "__main__":
    main()
