"""
Real Healthcare Dataset Integration

This script processes and integrates real healthcare datasets for use with
the personalization engine and explainability toolkit.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import yaml
import joblib
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealDatasetProcessor:
    """Process real healthcare datasets for ML analysis."""

    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processed_datasets = {}

        logger.info("🏥 Real Dataset Processor initialized")

    def process_heart_disease_uci(self) -> pd.DataFrame:
        """Process UCI Heart Disease dataset."""
        try:
            logger.info("Processing UCI Heart Disease dataset...")

            # Load the processed Cleveland data
            data_path = Path("data/heart+disease/processed.cleveland.data")

            if not data_path.exists():
                logger.warning(f"Heart disease data not found at {data_path}")
                return pd.DataFrame()

            # Column names from heart-disease.names
            columns = [
                "age",
                "sex",
                "cp",
                "trestbps",
                "chol",
                "fbs",
                "restecg",
                "thalach",
                "exang",
                "oldpeak",
                "slope",
                "ca",
                "thal",
                "target",
            ]

            # Read data
            df = pd.read_csv(data_path, names=columns, na_values="?")

            # Clean data
            # Convert target to binary (0: no disease, 1: disease)
            df["target"] = (df["target"] > 0).astype(int)

            # Add patient IDs
            df["patient_id"] = range(1, len(df) + 1)

            # Handle missing values
            df = df.fillna(df.median(numeric_only=True))

            # Add feature descriptions
            df.attrs["description"] = "UCI Heart Disease Dataset - Cleveland subset"
            df.attrs["target"] = "target"
            df.attrs["type"] = "classification"
            df.attrs["n_classes"] = 2

            logger.info(f"✅ Heart Disease dataset processed: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Failed to process heart disease dataset: {e}")
            return pd.DataFrame()

    def process_breast_cancer_wisconsin(self) -> pd.DataFrame:
        """Process Breast Cancer Wisconsin dataset."""
        try:
            logger.info("Processing Breast Cancer Wisconsin dataset...")

            data_path = Path("data/breast+cancer+wisconsin+diagnostic/wdbc.data")

            if not data_path.exists():
                logger.warning(f"Breast cancer data not found at {data_path}")
                return pd.DataFrame()

            # Column names from wdbc.names
            feature_names = [
                "radius",
                "texture",
                "perimeter",
                "area",
                "smoothness",
                "compactness",
                "concavity",
                "concave_points",
                "symmetry",
                "fractal_dimension",
            ]

            columns = ["id", "diagnosis"]
            # Add mean, SE, and worst for each feature
            for stat in ["mean", "se", "worst"]:
                for feature in feature_names:
                    columns.append(f"{feature}_{stat}")

            # Read data
            df = pd.read_csv(data_path, names=columns)

            # Convert diagnosis to binary
            df["target"] = (df["diagnosis"] == "M").astype(int)
            df["patient_id"] = df["id"]

            # Drop original columns
            df = df.drop(["id", "diagnosis"], axis=1)

            # Reorder columns
            cols = ["patient_id", "target"] + [
                col for col in df.columns if col not in ["patient_id", "target"]
            ]
            df = df[cols]

            df.attrs["description"] = "Breast Cancer Wisconsin (Diagnostic) Dataset"
            df.attrs["target"] = "target"
            df.attrs["type"] = "classification"
            df.attrs["n_classes"] = 2

            logger.info(f"✅ Breast Cancer dataset processed: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Failed to process breast cancer dataset: {e}")
            return pd.DataFrame()

    def process_diabetes_130_hospitals(self) -> pd.DataFrame:
        """Process Diabetes 130 US Hospitals dataset."""
        try:
            logger.info("Processing Diabetes 130 US Hospitals dataset...")

            data_path = Path(
                "data/diabetes+130-us+hospitals+for+years+1999-2008/diabetic_data.csv"
            )

            if not data_path.exists():
                logger.warning(f"Diabetes data not found at {data_path}")
                return pd.DataFrame()

            # Read data (sample first 5000 rows for manageable processing)
            df = pd.read_csv(data_path, nrows=5000)

            # Clean and preprocess
            # Handle age ranges - convert to numeric
            age_mapping = {
                "[0-10)": 5,
                "[10-20)": 15,
                "[20-30)": 25,
                "[30-40)": 35,
                "[40-50)": 45,
                "[50-60)": 55,
                "[60-70)": 65,
                "[70-80)": 75,
                "[80-90)": 85,
                "[90-100)": 95,
            }
            df["age_numeric"] = df["age"].map(age_mapping)

            # Convert gender to numeric
            df["gender_numeric"] = (df["gender"] == "Male").astype(int)

            # Convert race to numeric (simplified)
            race_mapping = {
                "Caucasian": 0,
                "AfricanAmerican": 1,
                "Hispanic": 2,
                "Other": 3,
                "Asian": 4,
                "?": 5,
            }
            df["race_numeric"] = df["race"].map(race_mapping)

            # Target variable - readmission
            target_mapping = {"NO": 0, "<30": 1, ">30": 1}
            df["target"] = df["readmitted"].map(target_mapping)

            # Select relevant features
            feature_cols = [
                "patient_nbr",
                "age_numeric",
                "gender_numeric",
                "race_numeric",
                "time_in_hospital",
                "num_lab_procedures",
                "num_procedures",
                "num_medications",
                "number_outpatient",
                "number_emergency",
                "number_inpatient",
                "number_diagnoses",
                "target",
            ]

            df_clean = df[feature_cols].copy()
            df_clean["patient_id"] = df_clean["patient_nbr"]
            df_clean = df_clean.drop("patient_nbr", axis=1)

            # Handle missing values
            df_clean = df_clean.fillna(df_clean.median(numeric_only=True))

            df_clean.attrs["description"] = "Diabetes 130 US Hospitals Dataset"
            df_clean.attrs["target"] = "target"
            df_clean.attrs["type"] = "classification"
            df_clean.attrs["n_classes"] = 2

            logger.info(f"✅ Diabetes dataset processed: {df_clean.shape}")
            return df_clean

        except Exception as e:
            logger.error(f"Failed to process diabetes dataset: {e}")
            return pd.DataFrame()

    def process_hepatitis_dataset(self) -> pd.DataFrame:
        """Process Hepatitis dataset."""
        try:
            logger.info("Processing Hepatitis dataset...")

            data_path = Path("data/hepatitis/hepatitis.data")

            if not data_path.exists():
                logger.warning(f"Hepatitis data not found at {data_path}")
                return pd.DataFrame()

            columns = [
                "class",
                "age",
                "sex",
                "steroid",
                "antivirals",
                "fatigue",
                "malaise",
                "anorexia",
                "liver_big",
                "liver_firm",
                "spleen_palpable",
                "spiders",
                "ascites",
                "varices",
                "bilirubin",
                "alk_phosphate",
                "sgot",
                "albumin",
                "protime",
                "histology",
            ]

            df = pd.read_csv(data_path, names=columns, na_values="?")

            # Convert target (1=die, 2=live to 0=die, 1=live)
            df["target"] = (df["class"] == 2).astype(int)
            df["patient_id"] = range(1, len(df) + 1)

            # Drop original class column
            df = df.drop("class", axis=1)

            # Handle missing values with median/mode
            for col in df.columns:
                if df[col].dtype in ["float64", "int64"]:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(
                        df[col].mode()[0] if len(df[col].mode()) > 0 else 0
                    )

            df.attrs["description"] = "Hepatitis Dataset"
            df.attrs["target"] = "target"
            df.attrs["type"] = "classification"
            df.attrs["n_classes"] = 2

            logger.info(f"✅ Hepatitis dataset processed: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Failed to process hepatitis dataset: {e}")
            return pd.DataFrame()

    def process_dermatology_dataset(self) -> pd.DataFrame:
        """Process Dermatology dataset."""
        try:
            logger.info("Processing Dermatology dataset...")

            data_path = Path("data/dermatology/dermatology.data")

            if not data_path.exists():
                logger.warning(f"Dermatology data not found at {data_path}")
                return pd.DataFrame()

            # Read data
            df = pd.read_csv(data_path, na_values="?")

            # The last column is the class (target)
            df.columns = [f"feature_{i}" for i in range(len(df.columns) - 1)] + [
                "target"
            ]

            # Convert target to 0-indexed
            df["target"] = df["target"] - 1
            df["patient_id"] = range(1, len(df) + 1)

            # Handle missing values
            df = df.fillna(df.median(numeric_only=True))

            df.attrs["description"] = "Dermatology Dataset"
            df.attrs["target"] = "target"
            df.attrs["type"] = "classification"
            df.attrs["n_classes"] = df["target"].nunique()

            logger.info(f"✅ Dermatology dataset processed: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Failed to process dermatology dataset: {e}")
            return pd.DataFrame()

    def process_medical_appointments(self) -> pd.DataFrame:
        """Process Medical Appointments No-Show dataset."""
        try:
            logger.info("Processing Medical Appointments No-Show dataset...")

            data_path = Path("data/Medical Appointments No-Show.csv")

            if not data_path.exists():
                logger.warning(f"Medical appointments data not found at {data_path}")
                return pd.DataFrame()

            df = pd.read_csv(data_path)

            # Clean column names
            df.columns = df.columns.str.replace("-", "_").str.lower()

            # Convert date columns
            if "scheduledday" in df.columns:
                df["scheduledday"] = pd.to_datetime(df["scheduledday"])
            if "appointmentday" in df.columns:
                df["appointmentday"] = pd.to_datetime(df["appointmentday"])

            # Create features
            if "scheduledday" in df.columns and "appointmentday" in df.columns:
                df["days_advance"] = (df["appointmentday"] - df["scheduledday"]).dt.days

            # Convert categorical variables
            if "gender" in df.columns:
                df["gender_numeric"] = (df["gender"] == "M").astype(int)

            # Target variable (No-show)
            if "no_show" in df.columns:
                df["target"] = (df["no_show"] == "Yes").astype(int)
            elif "noshow" in df.columns:
                df["target"] = (df["noshow"] == "Yes").astype(int)

            # Select relevant numeric features
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if "target" not in numeric_cols:
                numeric_cols.append("target")

            df_clean = df[numeric_cols].copy()

            # Add patient ID if not exists
            if "patient_id" not in df_clean.columns:
                if "patientid" in df_clean.columns:
                    df_clean["patient_id"] = df_clean["patientid"]
                else:
                    df_clean["patient_id"] = range(1, len(df_clean) + 1)

            # Handle missing values
            df_clean = df_clean.fillna(df_clean.median(numeric_only=True))

            df_clean.attrs["description"] = "Medical Appointments No-Show Dataset"
            df_clean.attrs["target"] = "target"
            df_clean.attrs["type"] = "classification"
            df_clean.attrs["n_classes"] = 2

            logger.info(f"✅ Medical Appointments dataset processed: {df_clean.shape}")
            return df_clean

        except Exception as e:
            logger.error(f"Failed to process medical appointments dataset: {e}")
            return pd.DataFrame()

    def process_covid_symptoms(self) -> pd.DataFrame:
        """Process COVID-19 Symptoms Survey dataset."""
        try:
            logger.info("Processing COVID-19 Symptoms Survey dataset...")

            data_path = Path("data/covid_19_symptoms_survey.csv")

            if not data_path.exists():
                logger.warning(f"COVID symptoms data not found at {data_path}")
                return pd.DataFrame()

            df = pd.read_csv(data_path)

            # Clean column names
            df.columns = df.columns.str.replace(" ", "_").str.lower()

            # Convert categorical variables to numeric
            for col in df.columns:
                if df[col].dtype == "object":
                    # Try to convert Yes/No to 1/0
                    if df[col].str.contains("Yes|No", na=False).any():
                        df[col] = (df[col] == "Yes").astype(int)
                    else:
                        # Use label encoding for other categorical variables
                        df[col] = pd.Categorical(df[col]).codes

            # Create target variable (assuming COVID positive/negative)
            covid_cols = [
                col
                for col in df.columns
                if "covid" in col.lower() or "positive" in col.lower()
            ]
            if covid_cols:
                df["target"] = df[covid_cols[0]]
            else:
                # Create synthetic target based on symptoms
                symptom_cols = [
                    col
                    for col in df.columns
                    if any(
                        symptom in col.lower()
                        for symptom in ["fever", "cough", "fatigue", "shortness"]
                    )
                ]
                if symptom_cols:
                    df["target"] = (df[symptom_cols].sum(axis=1) >= 2).astype(int)
                else:
                    df["target"] = np.random.choice([0, 1], size=len(df))

            # Add patient ID
            df["patient_id"] = range(1, len(df) + 1)

            # Handle missing values
            df = df.fillna(df.median(numeric_only=True))

            df.attrs["description"] = "COVID-19 Symptoms Survey Dataset"
            df.attrs["target"] = "target"
            df.attrs["type"] = "classification"
            df.attrs["n_classes"] = 2

            logger.info(f"✅ COVID Symptoms dataset processed: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Failed to process COVID symptoms dataset: {e}")
            return pd.DataFrame()

    def process_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Process all available real datasets."""
        logger.info("🔄 Processing all real healthcare datasets...")

        datasets = {
            "heart_disease_uci": self.process_heart_disease_uci(),
            "breast_cancer_wisconsin": self.process_breast_cancer_wisconsin(),
            "diabetes_130_hospitals": self.process_diabetes_130_hospitals(),
            "hepatitis": self.process_hepatitis_dataset(),
            "dermatology": self.process_dermatology_dataset(),
            "medical_appointments": self.process_medical_appointments(),
            "covid_symptoms": self.process_covid_symptoms(),
        }

        # Filter out empty datasets
        processed_datasets = {name: df for name, df in datasets.items() if not df.empty}

        # Save processed datasets
        for name, df in processed_datasets.items():
            save_path = self.output_dir / f"{name}_processed.csv"
            df.to_csv(save_path, index=False)
            logger.info(f"💾 Saved {name}: {save_path}")

        # Create summary
        summary = {
            "processed_datasets": list(processed_datasets.keys()),
            "dataset_info": {
                name: {
                    "shape": df.shape,
                    "target": df.attrs.get("target", "unknown"),
                    "type": df.attrs.get("type", "unknown"),
                    "n_classes": df.attrs.get("n_classes", "unknown"),
                    "description": df.attrs.get("description", "No description"),
                }
                for name, df in processed_datasets.items()
            },
            "timestamp": datetime.now().isoformat(),
        }

        # Save summary
        summary_path = self.output_dir / "dataset_summary.yaml"
        with open(summary_path, "w") as f:
            yaml.dump(summary, f, default_flow_style=False)

        logger.info(f"📋 Dataset summary saved: {summary_path}")
        logger.info(
            f"🎉 Processed {len(processed_datasets)} real datasets successfully!"
        )

        self.processed_datasets = processed_datasets
        return processed_datasets

    def get_dataset_for_personalization(
        self, dataset_name: str
    ) -> Optional[pd.DataFrame]:
        """Get a specific dataset formatted for personalization engine."""
        if dataset_name not in self.processed_datasets:
            logger.error(
                f"Dataset '{dataset_name}' not found. Available: {list(self.processed_datasets.keys())}"
            )
            return None

        df = self.processed_datasets[dataset_name].copy()

        # Ensure required columns exist
        if "patient_id" not in df.columns:
            df["patient_id"] = range(1, len(df) + 1)

        logger.info(
            f"✅ Dataset '{dataset_name}' ready for personalization: {df.shape}"
        )
        return df


def main():
    """Process all real healthcare datasets."""
    processor = RealDatasetProcessor()

    # Process all datasets
    datasets = processor.process_all_datasets()

    # Display summary
    print("\n" + "=" * 60)
    print("📊 REAL HEALTHCARE DATASETS PROCESSED")
    print("=" * 60)

    for name, df in datasets.items():
        print(f"\n🏥 {name.upper().replace('_', ' ')}")
        print(f"   Shape: {df.shape}")
        print(f"   Target: {df.attrs.get('target', 'unknown')}")
        print(f"   Type: {df.attrs.get('type', 'unknown')}")
        print(f"   Classes: {df.attrs.get('n_classes', 'unknown')}")
        print(f"   Description: {df.attrs.get('description', 'No description')}")

    print(f"\n🎉 All datasets saved to: data/processed/")
    print(f"📋 Summary available at: data/processed/dataset_summary.yaml")

    return datasets


if __name__ == "__main__":
    main()
