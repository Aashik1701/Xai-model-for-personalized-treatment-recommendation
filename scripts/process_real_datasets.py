"""
Real Healthcare Dataset Integration

This script processes and integrates real healthcare datasets for use with
the personalization engine and explainability toolkit.
"""

import pandas as pd  # type: ignore[import]
import numpy as np  # type: ignore[import]
import logging
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional
import yaml  # type: ignore[import]
from datetime import datetime
from PIL import Image  # type: ignore[import]

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MEDICAL_ONTOLOGY_CONCEPTS: List[Dict[str, object]] = [
    {
        "domain": "Cardiology",
        "code": "SNOMEDCT:49872002",
        "name": "Disorder of cardiovascular system",
        "keywords": {
            "cardio",
            "heart",
            "angina",
            "myocard",
            "cholesterol",
            "blood pressure",
            "hypertension",
            "tachycardia",
            "arrythmia",
        },
    },
    {
        "domain": "Endocrinology",
        "code": "SNOMEDCT:237602007",
        "name": "Endocrine system disorder",
        "keywords": {
            "diabet",
            "glucose",
            "insulin",
            "thyroid",
            "hormone",
            "metabolic",
        },
    },
    {
        "domain": "Dermatology",
        "code": "SNOMEDCT:23853001",
        "name": "Disorder of skin",
        "keywords": {
            "skin",
            "lesion",
            "rash",
            "derm",
            "acne",
            "melanoma",
            "keratosis",
        },
    },
    {
        "domain": "Infectious disease",
        "code": "SNOMEDCT:40733004",
        "name": "Communicable disease",
        "keywords": {
            "infection",
            "covid",
            "virus",
            "flu",
            "pneumonia",
            "antibiotic",
        },
    },
    {
        "domain": "Respiratory",
        "code": "SNOMEDCT:50043002",
        "name": "Disorder of respiratory system",
        "keywords": {
            "asthma",
            "breath",
            "respir",
            "lung",
            "bronch",
        },
    },
    {
        "domain": "Neurology",
        "code": "SNOMEDCT:118940003",
        "name": "Disorder of nervous system",
        "keywords": {
            "neuro",
            "brain",
            "stroke",
            "seizure",
            "dementia",
            "headache",
        },
    },
    {
        "domain": "Mental health",
        "code": "SNOMEDCT:74732009",
        "name": "Mental disorder",
        "keywords": {
            "depress",
            "anxiety",
            "mental",
            "stress",
            "psych",
            "bipolar",
        },
    },
    {
        "domain": "Gastroenterology",
        "code": "SNOMEDCT:118171006",
        "name": "Disorder of digestive system",
        "keywords": {
            "gastro",
            "stomach",
            "bowel",
            "abdomen",
            "reflux",
            "colon",
        },
    },
    {
        "domain": "Musculoskeletal",
        "code": "SNOMEDCT:928000",
        "name": "Disorder of musculoskeletal system",
        "keywords": {
            "bone",
            "joint",
            "muscle",
            "arthritis",
            "back pain",
        },
    },
]


DRUG_RXNORM_LOOKUP: Dict[str, str] = {
    "atorvastatin": "RXNORM:83367",
    "metformin": "RXNORM:860975",
    "lisinopril": "RXNORM:29046",
    "omeprazole": "RXNORM:9678",
    "ibuprofen": "RXNORM:5640",
    "levothyroxine": "RXNORM:966249",
    "insulin": "RXNORM:847232",
    "amoxicillin": "RXNORM:723",
    "sertraline": "RXNORM:865098",
    "acetaminophen": "RXNORM:161",
}


LESION_ONTOLOGY_MAP: Dict[str, Dict[str, str]] = {
    "actinic keratosis": {"icd10": "L57.0", "risk": "premalignant"},
    "basal cell carcinoma": {"icd10": "C44.91", "risk": "malignant"},
    "dermatofibroma": {"icd10": "D23.5", "risk": "benign"},
    "melanoma": {"icd10": "C43.9", "risk": "malignant"},
    "nevus": {"icd10": "D22.9", "risk": "benign"},
    "pigmented benign keratosis": {"icd10": "L82.1", "risk": "benign"},
    "seborrheic keratosis": {"icd10": "L82.1", "risk": "benign"},
    "squamous cell carcinoma": {"icd10": "C44.92", "risk": "malignant"},
    "vascular lesion": {"icd10": "I78.1", "risk": "benign"},
}


def map_text_to_concept(text: str) -> Dict[str, str]:
    """Map free text to a lightweight clinical ontology concept."""

    if not isinstance(text, str) or not text.strip():
        return {
            "code": "SNOMEDCT:000000",
            "name": "Unknown concept",
            "domain": "General",
        }

    normalized = text.lower()
    for concept in MEDICAL_ONTOLOGY_CONCEPTS:
        keywords = concept["keywords"]  # type: ignore[index]
        if any(keyword in normalized for keyword in keywords):
            return {
                "code": concept["code"],  # type: ignore[index]
                "name": concept["name"],  # type: ignore[index]
                "domain": concept["domain"],  # type: ignore[index]
            }

    return {
        "code": "SNOMEDCT:000000",
        "name": "Unknown concept",
        "domain": "General",
    }


def summarise_list(values: List[str], limit: int = 5) -> str:
    """Generate a pipe-delimited summary for top-N values in a collection."""

    filtered = [value for value in values if isinstance(value, str) and value]
    if not filtered:
        return ""

    truncated = filtered[:limit]
    overflow = len(filtered) - len(truncated)
    summary = " | ".join(truncated)
    if overflow > 0:
        summary = f"{summary} | (+{overflow} more)"
    return summary


class RealDatasetProcessor:
    """Process real healthcare datasets for ML analysis."""

    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processed_datasets: Dict[str, pd.DataFrame] = {}

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

    def process_heart_disease_multicenter(self) -> pd.DataFrame:
        """Process multiple heart disease cohorts into a unified dataset."""
        try:
            logger.info("Processing multi-center Heart Disease cohorts...")

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

            cohort_files = {
                "cleveland": "processed.cleveland.data",
                "hungarian": "processed.hungarian.data",
                "switzerland": "processed.switzerland.data",
                "va": "processed.va.data",
            }

            cohort_frames: List[pd.DataFrame] = []

            for site, filename in cohort_files.items():
                data_path = Path("data/heart+disease") / filename
                if not data_path.exists():
                    logger.warning(
                        "Heart disease cohort missing (%s): %s", site, data_path
                    )
                    continue

                df_site = pd.read_csv(data_path, names=columns, na_values="?")

                if df_site.empty:
                    logger.warning(f"Heart disease cohort empty ({site}): {data_path}")
                    continue

                df_site["target"] = (df_site["target"] > 0).astype(int)
                df_site["site"] = site
                cohort_frames.append(df_site)

            if not cohort_frames:
                logger.warning(
                    "No heart disease cohorts processed; returning empty DataFrame"
                )
                return pd.DataFrame()

            df = pd.concat(cohort_frames, ignore_index=True)
            df["patient_id"] = range(1, len(df) + 1)
            df["site_numeric"] = pd.factorize(df["site"])[0]

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

            df.attrs["description"] = (
                "Multi-center Heart Disease Cohorts "
                "(Cleveland, Hungarian, Switzerland, VA)"
            )
            df.attrs["target"] = "target"
            df.attrs["type"] = "classification"
            df.attrs["n_classes"] = 2
            df.attrs["sites"] = sorted(df["site"].unique())

            logger.info(f"✅ Multi-center Heart Disease dataset processed: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Failed to process multi-center heart disease dataset: {e}")
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
                "admission_type_id",
                "discharge_disposition_id",
                "admission_source_id",
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

            # Ensure ID columns are numeric
            for col in [
                "admission_type_id",
                "discharge_disposition_id",
                "admission_source_id",
            ]:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

            # Map lookup descriptions for personalization narratives
            mapping_path = Path(
                "data/diabetes+130-us+hospitals+for+years+1999-2008/IDS_mapping.csv"
            )
            id_mappings = self._load_diabetes_lookup(mapping_path)
            for col_name, mapping in id_mappings.items():
                normalized_name = col_name.strip()
                if normalized_name in df_clean.columns and mapping:
                    desc_col = f"{normalized_name}_desc"
                    df_clean[desc_col] = (
                        df_clean[normalized_name]
                        .map(mapping)
                        .fillna("Unknown")
                        .astype(str)
                    )

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
                        df[col] = pd.factorize(df[col])[0]

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

    def process_thyroid_disease(self) -> pd.DataFrame:
        """Process Thyroid Disease (new-thyroid) dataset."""
        try:
            logger.info("Processing Thyroid Disease dataset (new-thyroid)...")

            data_path = Path("data/thyroid+disease/new-thyroid.data")

            if not data_path.exists():
                logger.warning(f"Thyroid disease data not found at {data_path}")
                return pd.DataFrame()

            columns = [
                "target",
                "t3_resin_uptake",
                "total_serum_thyroxin",
                "total_serum_triiodothyronine",
                "basal_tsh",
                "max_tsh_difference",
            ]

            df = pd.read_csv(data_path, names=columns, na_values=["?", "NA"])

            # Convert to numeric and handle missing values
            for col in columns[1:]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.fillna(df.median(numeric_only=True))

            # Map class labels (1=normal, 2=hyper, 3=hypo) to 0-indexed classes
            df["target"] = df["target"].astype(int) - 1

            # Add patient identifiers
            df["patient_id"] = range(1, len(df) + 1)

            df.attrs["description"] = "Thyroid Disease Dataset (new-thyroid)"
            df.attrs["target"] = "target"
            df.attrs["type"] = "classification"
            df.attrs["n_classes"] = int(df["target"].nunique())

            logger.info(f"✅ Thyroid Disease dataset processed: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Failed to process thyroid disease dataset: {e}")
            return pd.DataFrame()

    def process_synthea_longitudinal(self) -> pd.DataFrame:
        """Aggregate Synthea synthetic EHR into longitudinal patient features."""

        try:
            logger.info("Processing Synthea longitudinal corpus...")

            base_path = Path("data/Synthea Synthetic")
            if not base_path.exists():
                logger.warning("Synthea directory not found at %s", base_path)
                return pd.DataFrame()

            required_files = [
                "patients.csv",
                "encounters.csv",
                "conditions.csv",
                "medications.csv",
                "observations.csv",
                "procedures.csv",
                "careplans.csv",
                "immunizations.csv",
            ]

            missing_files = [
                filename
                for filename in required_files
                if not (base_path / filename).exists()
            ]
            if missing_files:
                logger.warning("Synthea files missing: %s", ", ".join(missing_files))
                return pd.DataFrame()

            def _load_csv(
                name: str, parse_dates: Optional[List[str]] = None
            ) -> pd.DataFrame:
                csv_path = base_path / name
                try:
                    return pd.read_csv(
                        csv_path, parse_dates=parse_dates, low_memory=False
                    )
                except Exception as exc:
                    logger.error("Failed to load %s: %s", csv_path, exc)
                    return pd.DataFrame()

            patients = _load_csv("patients.csv", parse_dates=["BIRTHDATE", "DEATHDATE"])
            encounters = _load_csv("encounters.csv", parse_dates=["START", "STOP"])
            conditions = _load_csv("conditions.csv", parse_dates=["START", "STOP"])
            medications = _load_csv("medications.csv", parse_dates=["START", "STOP"])
            observations = _load_csv("observations.csv", parse_dates=["DATE"])
            procedures = _load_csv("procedures.csv", parse_dates=["DATE"])
            careplans = _load_csv("careplans.csv", parse_dates=["START", "STOP"])
            immunizations = _load_csv("immunizations.csv", parse_dates=["DATE"])

            if patients.empty or encounters.empty:
                logger.warning(
                    "Insufficient Synthea data to build longitudinal dataset"
                )
                return pd.DataFrame()

            encounters["duration_days"] = (
                encounters["STOP"] - encounters["START"]
            ).dt.total_seconds() / 86400
            encounters["duration_days"] = encounters["duration_days"].fillna(0.0)
            encounters["is_emergency"] = (
                encounters["ENCOUNTERCLASS"].astype(str).str.lower().eq("emergency")
            )

            encounter_group = encounters.groupby("PATIENT")
            logger.info("✓ Loaded CSVs, building encounter aggregations...")
            encounter_agg = pd.DataFrame(
                {
                    "n_encounters": encounter_group.size(),
                    "unique_encounter_classes": encounter_group[
                        "ENCOUNTERCLASS"
                    ].nunique(),
                    "first_encounter_start": encounter_group["START"].min(),
                    "last_encounter_end": encounter_group["STOP"].max(),
                    "avg_encounter_duration_days": encounter_group[
                        "duration_days"
                    ].mean(),
                    "median_encounter_duration_days": encounter_group[
                        "duration_days"
                    ].median(),
                    "emergency_encounter_count": encounter_group["is_emergency"].sum(),
                }
            )
            logger.info("✓ Encounter aggregation complete")

            for col in [
                "avg_encounter_duration_days",
                "median_encounter_duration_days",
            ]:
                encounter_agg[col] = encounter_agg[col].fillna(0.0)

            logger.info("✓ Building condition aggregations...")
            conditions["is_chronic"] = conditions["STOP"].isna()
            condition_group = conditions.groupby("PATIENT")
            condition_agg = pd.DataFrame(
                {
                    "n_conditions": condition_group.size(),
                    "n_unique_condition_codes": condition_group["CODE"].nunique(),
                    "n_chronic_conditions": condition_group["is_chronic"].sum(),
                }
            )
            logger.info("✓ Condition counts complete")

            condition_descriptions = conditions.groupby("PATIENT")["DESCRIPTION"].apply(
                lambda series: summarise_list(series.dropna().tolist(), limit=5)
            )
            logger.info("✓ Condition descriptions summarized")

            condition_domains = conditions[["PATIENT", "DESCRIPTION"]].copy()
            condition_domains["concept_domain"] = condition_domains[
                "DESCRIPTION"
            ].apply(lambda desc: map_text_to_concept(str(desc))["domain"])

            def _most_common_domain(domains: List[str]) -> str:
                filtered = [dom for dom in domains if dom and dom != "General"]
                if not filtered:
                    return "General"
                return Counter(filtered).most_common(1)[0][0]

            dominant_domains = (
                condition_domains.groupby("PATIENT")["concept_domain"]
                .apply(list)
                .apply(_most_common_domain)
            )
            logger.info("✓ Dominant domains extracted")

            logger.info("✓ Building medication aggregations...")
            medications["is_active"] = medications["STOP"].isna()
            medication_group = medications.groupby("PATIENT")
            medication_agg = pd.DataFrame(
                {
                    "n_medications": medication_group.size(),
                    "n_unique_medication_codes": medication_group["CODE"].nunique(),
                    "n_active_medications": medication_group["is_active"].sum(),
                }
            )
            medication_descriptions = medications.groupby("PATIENT")[
                "DESCRIPTION"
            ].apply(lambda series: summarise_list(series.dropna().tolist(), limit=5))
            logger.info("✓ Medication aggregations complete")

            logger.info("✓ Processing observations...")

            observations["VALUE_NUMERIC"] = pd.to_numeric(
                observations.get("VALUE"), errors="coerce"
            )
            observation_numeric_counts = observations.groupby("PATIENT")[
                "VALUE_NUMERIC"
            ].apply(lambda series: series.notna().sum())
            logger.info("✓ Observation numeric values processed")

            logger.info("✓ Extracting key observation codes...")
            key_observation_codes = {
                "8480-6": "systolic_bp",
                "8462-4": "diastolic_bp",
                "2085-9": "hdl_cholesterol",
                "2089-1": "ldl_cholesterol",
                "2345-7": "glucose",
            }

            observation_features: Dict[str, pd.DataFrame] = {}
            for code, feature_name in key_observation_codes.items():
                subset = observations[observations["CODE"] == code].copy()
                if subset.empty:
                    continue
                subset_sorted = subset.sort_values("DATE")
                agg = subset_sorted.groupby("PATIENT")["VALUE_NUMERIC"].agg(
                    ["mean", "max", "min", "last"]
                )
                agg = agg.rename(
                    columns={
                        "mean": f"{feature_name}_mean",
                        "max": f"{feature_name}_max",
                        "min": f"{feature_name}_min",
                        "last": f"{feature_name}_last",
                    }
                )
                observation_features[feature_name] = agg
            logger.info("✓ Key observation features extracted")

            logger.info("✓ Building event timeline...")
            timeline_frames: List[pd.DataFrame] = []

            if not encounters.empty:
                timeline_frames.append(
                    encounters.assign(
                        event_date=encounters["START"],
                        event_type="encounter",
                        event_label=encounters["ENCOUNTERCLASS"].fillna("unknown"),
                    )[["PATIENT", "event_date", "event_type", "event_label"]]
                )

            if not conditions.empty:
                timeline_frames.append(
                    conditions.assign(
                        event_date=conditions["START"],
                        event_type="condition",
                        event_label=conditions["DESCRIPTION"].fillna("unknown"),
                    )[["PATIENT", "event_date", "event_type", "event_label"]]
                )

            if not medications.empty:
                timeline_frames.append(
                    medications.assign(
                        event_date=medications["START"],
                        event_type="medication",
                        event_label=medications["DESCRIPTION"].fillna("unknown"),
                    )[["PATIENT", "event_date", "event_type", "event_label"]]
                )

            if not procedures.empty:
                timeline_frames.append(
                    procedures.assign(
                        event_date=procedures["DATE"],
                        event_type="procedure",
                        event_label=procedures["DESCRIPTION"].fillna("unknown"),
                    )[["PATIENT", "event_date", "event_type", "event_label"]]
                )

            if not careplans.empty:
                timeline_frames.append(
                    careplans.assign(
                        event_date=careplans["START"],
                        event_type="careplan",
                        event_label=careplans["DESCRIPTION"].fillna("unknown"),
                    )[["PATIENT", "event_date", "event_type", "event_label"]]
                )

            if not immunizations.empty:
                timeline_frames.append(
                    immunizations.assign(
                        event_date=immunizations["DATE"],
                        event_type="immunization",
                        event_label=immunizations["DESCRIPTION"].fillna("unknown"),
                    )[["PATIENT", "event_date", "event_type", "event_label"]]
                )

            timeline_map: Dict[str, List[Dict[str, str]]] = {}
            event_counts = pd.Series(dtype=float)
            if timeline_frames:
                timeline_df = pd.concat(timeline_frames, ignore_index=True)
                timeline_df = timeline_df.dropna(subset=["event_date"])

                # Normalize all event_date to timezone-naive to avoid comparison errors
                timeline_df["event_date"] = pd.to_datetime(
                    timeline_df["event_date"], utc=True
                ).dt.tz_localize(None)

                timeline_df = timeline_df.sort_values(["PATIENT", "event_date"])
                event_counts = timeline_df.groupby("PATIENT").size()
                for patient_id, group in timeline_df.groupby("PATIENT"):
                    limited = group.head(50)
                    timeline_map[str(patient_id)] = [
                        {
                            "date": pd.to_datetime(row.event_date).isoformat(),
                            "type": str(row.event_type),
                            "label": str(row.event_label),
                        }
                        for row in limited.itertuples()
                        if pd.notna(row.event_date)
                    ]
            logger.info("✓ Event timeline complete")

            patient_df = patients[
                [
                    "Id",
                    "BIRTHDATE",
                    "DEATHDATE",
                    "RACE",
                    "ETHNICITY",
                    "GENDER",
                    "ZIP",
                    "CITY",
                    "STATE",
                    "MARITAL",
                ]
            ].copy()
            patient_df = patient_df.rename(columns={"Id": "patient_id"})
            logger.info("✓ Patient base extracted, starting merge...")

            longitudinal = patient_df.merge(
                encounter_agg, how="left", left_on="patient_id", right_index=True
            )
            logger.info("✓ Encounter merge complete")

            longitudinal = longitudinal.merge(
                condition_agg, how="left", left_on="patient_id", right_index=True
            )
            logger.info("✓ Condition merge complete")

            longitudinal = longitudinal.merge(
                medication_agg, how="left", left_on="patient_id", right_index=True
            )
            logger.info("✓ Medication merge complete")

            logger.info("✓ Adding summary fields...")
            longitudinal["condition_summary"] = (
                longitudinal["patient_id"].map(condition_descriptions).fillna("")
            )
            longitudinal["medication_summary"] = (
                longitudinal["patient_id"].map(medication_descriptions).fillna("")
            )
            longitudinal["dominant_condition_domain"] = (
                longitudinal["patient_id"].map(dominant_domains).fillna("General")
            )

            longitudinal["numeric_observation_count"] = (
                longitudinal["patient_id"].map(observation_numeric_counts).fillna(0)
            )

            for feature_df in observation_features.values():
                longitudinal = longitudinal.merge(
                    feature_df,
                    how="left",
                    left_on="patient_id",
                    right_index=True,
                )
            logger.info("✓ Observation features merged")

            logger.info("✓ Adding temporal event summary...")
            longitudinal["temporal_event_count"] = (
                longitudinal["patient_id"].map(event_counts).fillna(0).astype(int)
            )
            longitudinal["temporal_event_summary"] = longitudinal["patient_id"].map(
                lambda pid: json.dumps(
                    timeline_map.get(str(pid), []), ensure_ascii=False
                )
            )
            logger.info("✓ Temporal event summary complete")

            logger.info("✓ Calculating date features...")
            # Normalize all datetime columns to timezone-naive
            longitudinal["first_encounter_start"] = pd.to_datetime(
                longitudinal["first_encounter_start"], errors="coerce", utc=True
            ).dt.tz_localize(None)
            longitudinal["last_encounter_end"] = pd.to_datetime(
                longitudinal["last_encounter_end"], errors="coerce", utc=True
            ).dt.tz_localize(None)

            longitudinal["encounter_time_span_days"] = (
                (
                    longitudinal["last_encounter_end"]
                    - longitudinal["first_encounter_start"]
                )
                .dt.days.clip(lower=0)
                .fillna(0)
            )

            reference_date = longitudinal["last_encounter_end"].fillna(
                pd.Timestamp(datetime.now())
            )
            birthdate = pd.to_datetime(
                longitudinal["BIRTHDATE"], errors="coerce", utc=True
            ).dt.tz_localize(None)
            longitudinal["age_at_last_encounter_years"] = (
                ((reference_date - birthdate).dt.days / 365.25).fillna(0).round(1)
            )
            longitudinal["has_death_date"] = (
                longitudinal["DEATHDATE"].notna().astype(int)
            )
            logger.info("✓ Date features calculated")

            logger.info("✓ Encoding categorical variables...")
            longitudinal["gender_code"] = (
                longitudinal["GENDER"]
                .fillna("Unknown")
                .str.upper()
                .map({"M": 1, "F": 0})
                .fillna(0)
            ).astype(int)
            longitudinal["race_code"] = pd.factorize(
                longitudinal["RACE"].fillna("Unknown")
            )[0]
            longitudinal["ethnicity_code"] = pd.factorize(
                longitudinal["ETHNICITY"].fillna("Unknown")
            )[0]
            logger.info("✓ Categorical encoding complete")

            logger.info("✓ Computing derived metrics...")
            longitudinal["longitudinal_complexity_score"] = np.log1p(
                longitudinal["n_encounters"].fillna(0)
                + longitudinal["n_chronic_conditions"].fillna(0) * 1.5
                + longitudinal["n_active_medications"].fillna(0)
            ).round(3)

            longitudinal["target"] = (
                (longitudinal["n_encounters"].fillna(0) >= 5)
                | (longitudinal["n_chronic_conditions"].fillna(0) >= 3)
                | (longitudinal["n_active_medications"].fillna(0) >= 4)
                | (longitudinal["encounter_time_span_days"] >= 365)
            ).astype(int)

            longitudinal["birthdate"] = birthdate.dt.date.astype(str)
            longitudinal["deathdate"] = pd.to_datetime(
                longitudinal["DEATHDATE"], errors="coerce"
            ).dt.date.astype(str)
            logger.info("✓ Derived metrics complete")

            columns_order = [
                "patient_id",
                "target",
                "age_at_last_encounter_years",
                "gender_code",
                "race_code",
                "ethnicity_code",
                "n_encounters",
                "unique_encounter_classes",
                "emergency_encounter_count",
                "encounter_time_span_days",
                "avg_encounter_duration_days",
                "median_encounter_duration_days",
                "n_conditions",
                "n_unique_condition_codes",
                "n_chronic_conditions",
                "n_medications",
                "n_unique_medication_codes",
                "n_active_medications",
                "numeric_observation_count",
                "longitudinal_complexity_score",
                "temporal_event_count",
            ]

            observation_feature_columns = [
                col
                for col in longitudinal.columns
                if any(
                    col.startswith(prefix) for prefix in key_observation_codes.values()
                )
            ]

            extra_columns = [
                "birthdate",
                "deathdate",
                "GENDER",
                "RACE",
                "ETHNICITY",
                "ZIP",
                "CITY",
                "STATE",
                "MARITAL",
                "condition_summary",
                "medication_summary",
                "dominant_condition_domain",
                "temporal_event_summary",
            ]

            ordered_columns = (
                columns_order + observation_feature_columns + extra_columns
            )
            ordered_columns = [
                col for col in ordered_columns if col in longitudinal.columns
            ]
            logger.info("✓ Reordering columns...")
            longitudinal = longitudinal[ordered_columns].copy()
            logger.info("✓ Column reordering complete")

            longitudinal.attrs["description"] = (
                "Synthea synthetic longitudinal patient summary with temporal metrics, "
                "condition/medication ontology context, and high-utilization target"
            )
            longitudinal.attrs["target"] = "target"
            longitudinal.attrs["type"] = "classification"
            longitudinal.attrs["n_classes"] = 2

            logger.info(
                "✅ Synthea longitudinal dataset processed: %s", longitudinal.shape
            )
            return longitudinal

        except Exception as exc:
            logger.error("Failed to process Synthea longitudinal dataset: %s", exc)
            import traceback

            traceback.print_exc()
            return pd.DataFrame()

    def process_medical_question_pairs(self) -> pd.DataFrame:
        """Process Medical Question Duplicate Detection dataset."""

        try:
            logger.info("Processing Medical Question Pairs dataset...")

            data_path = Path("data/medical_questions_pairs.csv")

            if not data_path.exists():
                logger.warning("Medical question pairs data not found at %s", data_path)
                return pd.DataFrame()

            columns = [
                "pair_group_id",
                "question_primary",
                "question_secondary",
                "is_duplicate",
            ]

            df = pd.read_csv(data_path, names=columns)

            df["question_primary"] = (
                df["question_primary"].astype(str).fillna("").str.strip()
            )
            df["question_secondary"] = (
                df["question_secondary"].astype(str).fillna("").str.strip()
            )

            df["pair_group_id"] = pd.to_numeric(
                df["pair_group_id"], errors="coerce"
            ).fillna(-1)
            df["pair_group_id"] = df["pair_group_id"].astype(int)

            df["is_duplicate"] = pd.to_numeric(
                df["is_duplicate"], errors="coerce"
            ).fillna(0)
            df["is_duplicate"] = df["is_duplicate"].astype(int)
            df["duplicate_label"] = df["is_duplicate"].map(
                {1: "duplicate", 0: "distinct"}
            )

            for prefix, column in [
                ("primary", "question_primary"),
                ("secondary", "question_secondary"),
            ]:
                df[f"{prefix}_char_length"] = df[column].str.len()
                df[f"{prefix}_word_count"] = (
                    df[column].str.split().str.len().fillna(0).astype(int)
                )

            primary_len = df["primary_char_length"].clip(lower=1)
            secondary_len = df["secondary_char_length"].clip(lower=1)
            df["length_ratio"] = np.minimum(primary_len, secondary_len) / np.maximum(
                primary_len, secondary_len
            )

            def compute_overlap(row: pd.Series) -> float:
                tokens_a = set(row["question_primary"].lower().split())
                tokens_b = set(row["question_secondary"].lower().split())
                if not tokens_a or not tokens_b:
                    return 0.0
                overlap = len(tokens_a & tokens_b)
                union = len(tokens_a | tokens_b)
                return overlap / union if union else 0.0

            df["token_jaccard"] = df.apply(compute_overlap, axis=1)

            primary_concepts = df["question_primary"].apply(map_text_to_concept)
            df["primary_concept_code"] = primary_concepts.apply(lambda c: c["code"])
            df["primary_concept_name"] = primary_concepts.apply(lambda c: c["name"])
            df["primary_concept_domain"] = primary_concepts.apply(lambda c: c["domain"])

            secondary_concepts = df["question_secondary"].apply(map_text_to_concept)
            df["secondary_concept_code"] = secondary_concepts.apply(lambda c: c["code"])
            df["secondary_concept_name"] = secondary_concepts.apply(lambda c: c["name"])
            df["secondary_concept_domain"] = secondary_concepts.apply(
                lambda c: c["domain"]
            )

            df["concept_alignment"] = (
                df["primary_concept_code"] == df["secondary_concept_code"]
            ).astype(int)
            df["has_ontology_match"] = (
                (df["primary_concept_code"] != "SNOMEDCT:000000")
                | (df["secondary_concept_code"] != "SNOMEDCT:000000")
            ).astype(int)
            df["shared_concept_domain"] = np.where(
                df["concept_alignment"] == 1,
                df["primary_concept_domain"],
                "Mixed/Unaligned",
            )
            df["concept_pair_summary"] = (
                df["primary_concept_domain"].str.title()
                + " ↔ "
                + df["secondary_concept_domain"].str.title()
            )

            df.attrs["description"] = (
                "Medical question duplicate detection pairs with engineered "
                "length and token-overlap features"
            )
            df.attrs["target"] = "is_duplicate"
            df.attrs["type"] = "classification"
            df.attrs["n_classes"] = int(df["is_duplicate"].nunique())

            logger.info("✅ Medical Question Pairs dataset processed: %s", df.shape)
            return df

        except Exception as exc:
            logger.error("Failed to process medical question pairs dataset: %s", exc)
            return pd.DataFrame()

    def process_drug_reviews(self) -> pd.DataFrame:
        """Process DrugLib review dataset (train + test)."""

        try:
            logger.info("Processing Drug Review dataset (DrugLib)...")

            base_path = Path("data/drug+review+dataset+druglib")
            split_files = {
                "train": "drugLibTrain_raw.tsv",
                "test": "drugLibTest_raw.tsv",
            }

            frames: List[pd.DataFrame] = []

            for split, filename in split_files.items():
                file_path = base_path / filename
                if not file_path.exists():
                    logger.warning("Drug review %s split missing: %s", split, file_path)
                    continue

                df_split = pd.read_csv(file_path, sep="\t", engine="python")
                df_split["dataset_split"] = split
                frames.append(df_split)

            if not frames:
                logger.warning(
                    "No drug review splits processed; returning empty DataFrame"
                )
                return pd.DataFrame()

            df = pd.concat(frames, ignore_index=True)

            rename_map = {
                "Unnamed: 0": "review_id",
                "urlDrugName": "drug_name",
                "sideEffects": "side_effects",
                "benefitsReview": "benefits_review",
                "sideEffectsReview": "side_effects_review",
                "commentsReview": "comments_review",
            }
            df = df.rename(columns=rename_map)

            text_columns = [
                "benefits_review",
                "side_effects_review",
                "comments_review",
            ]
            for column in text_columns:
                df[column] = df[column].fillna("").astype(str).str.strip()

            df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
            df = df.dropna(subset=["rating"])
            df["rating"] = df["rating"].astype(int)

            effectiveness_map = {
                "Highly Effective": 5,
                "Considerably Effective": 4,
                "Moderately Effective": 3,
                "Marginally Effective": 2,
                "Ineffective": 1,
            }

            side_effect_map = {
                "No Side Effects": 1,
                "Mild Side Effects": 2,
                "Moderate Side Effects": 3,
                "Severe Side Effects": 4,
                "Extremely Severe Side Effects": 5,
            }

            df["effectiveness_score"] = df["effectiveness"].map(effectiveness_map)
            df["side_effect_severity"] = df["side_effects"].map(side_effect_map)

            for column in text_columns:
                prefix = column.replace("_review", "")
                df[f"{prefix}_char_length"] = df[column].str.len()
                df[f"{prefix}_word_count"] = (
                    df[column].str.split().str.len().fillna(0).astype(int)
                )

            df["total_review_word_count"] = (
                df[[f"{col.replace('_review', '')}_word_count" for col in text_columns]]
                .sum(axis=1)
                .astype(int)
            )

            df["net_rating_vs_side_effects"] = df["rating"] - df[
                "side_effect_severity"
            ].fillna(0)

            df["drug_name"] = df["drug_name"].fillna("Unknown").astype(str).str.strip()
            df["condition"] = df["condition"].fillna("Unknown").astype(str).str.strip()

            df["drug_rxnorm_code"] = (
                df["drug_name"]
                .str.lower()
                .map(DRUG_RXNORM_LOOKUP)
                .fillna("RXNORM:UNKNOWN")
            )

            condition_concepts = df["condition"].apply(map_text_to_concept)
            df["condition_concept_code"] = condition_concepts.apply(lambda c: c["code"])
            df["condition_concept_name"] = condition_concepts.apply(lambda c: c["name"])
            df["condition_concept_domain"] = condition_concepts.apply(
                lambda c: c["domain"]
            )
            df["condition_has_ontology_match"] = (
                df["condition_concept_code"] != "SNOMEDCT:000000"
            ).astype(int)

            df["effectiveness_minus_side_effects"] = df["effectiveness_score"].fillna(
                0
            ) - df["side_effect_severity"].fillna(0)

            df.attrs["description"] = (
                "DrugLib patient reviews with split indicator, text-length features, "
                "and ordinal encodings for effectiveness and side-effect severity"
            )
            df.attrs["target"] = "rating"
            df.attrs["type"] = "regression"
            df.attrs["n_classes"] = "continuous"

            logger.info("✅ Drug Review dataset processed: %s", df.shape)
            return df

        except Exception as exc:
            logger.error("Failed to process drug review dataset: %s", exc)
            return pd.DataFrame()

    def process_skin_cancer_imaging_metadata(self) -> pd.DataFrame:
        """Extract metadata features from the ISIC skin cancer sample."""

        try:
            logger.info("Processing ISIC skin cancer imaging metadata...")

            base_path = Path("data/isic_skin_cancer_sample")
            if not base_path.exists():
                logger.warning("ISIC sample directory not found at %s", base_path)
                return pd.DataFrame()

            supported_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
            records: List[Dict[str, object]] = []

            for split_dir in base_path.iterdir():
                if not split_dir.is_dir() or split_dir.name.startswith("."):
                    continue
                split_name = split_dir.name.lower()

                for label_dir in split_dir.iterdir():
                    if not label_dir.is_dir() or label_dir.name.startswith("."):
                        continue

                    lesion_label = label_dir.name.strip()
                    label_key = lesion_label.lower()
                    ontology_info = LESION_ONTOLOGY_MAP.get(
                        label_key,
                        {"icd10": "UNKNOWN", "risk": "unknown"},
                    )
                    is_malignant = int(ontology_info.get("risk") == "malignant")

                    for image_path in label_dir.glob("**/*"):
                        if not image_path.is_file():
                            continue
                        if image_path.suffix.lower() not in supported_extensions:
                            continue

                        try:
                            with Image.open(image_path) as img:
                                img_rgb = img.convert("RGB")
                                width, height = img_rgb.size
                                aspect_ratio = round(width / height, 4) if height else 0
                                area = width * height

                                rgb_array = np.array(img_rgb)
                                mean_r = float(rgb_array[:, :, 0].mean())
                                mean_g = float(rgb_array[:, :, 1].mean())
                                mean_b = float(rgb_array[:, :, 2].mean())
                                std_r = float(rgb_array[:, :, 0].std())
                                std_g = float(rgb_array[:, :, 1].std())
                                std_b = float(rgb_array[:, :, 2].std())

                                gray_array = np.array(img_rgb.convert("L"))
                                mean_intensity = float(gray_array.mean())
                                std_intensity = float(gray_array.std())
                        except Exception as exc:  # pragma: no cover - corrupt images
                            logger.debug(
                                "Skipping image %s due to error: %s", image_path, exc
                            )
                            continue

                        try:
                            relative_path = image_path.relative_to(Path.cwd())
                        except ValueError:
                            relative_path = image_path

                        records.append(
                            {
                                "image_path": str(relative_path),
                                "split": split_name,
                                "lesion_label": lesion_label,
                                "lesion_label_clean": label_key.replace(" ", "_"),
                                "lesion_icd10_code": ontology_info.get(
                                    "icd10", "UNKNOWN"
                                ),
                                "lesion_risk_category": ontology_info.get(
                                    "risk", "unknown"
                                ),
                                "is_malignant": is_malignant,
                                "target": is_malignant,
                                "width": width,
                                "height": height,
                                "aspect_ratio": aspect_ratio,
                                "image_area": area,
                                "mean_intensity": round(mean_intensity, 4),
                                "std_intensity": round(std_intensity, 4),
                                "mean_red": round(mean_r, 4),
                                "mean_green": round(mean_g, 4),
                                "mean_blue": round(mean_b, 4),
                                "std_red": round(std_r, 4),
                                "std_green": round(std_g, 4),
                                "std_blue": round(std_b, 4),
                            }
                        )

            if not records:
                logger.warning(
                    "No ISIC imaging records processed — verify dataset extraction."
                )
                return pd.DataFrame()

            df = pd.DataFrame(records)
            df["lesion_class_index"] = pd.factorize(df["lesion_label"])[0]
            df["concept_domain"] = "Dermatology"

            df.attrs["description"] = (
                "ISIC skin cancer image metadata with ICD-10 ontology tags, "
                "malignancy labels, and basic computer-vision summary features"
            )
            df.attrs["target"] = "target"
            df.attrs["type"] = "classification"
            df.attrs["n_classes"] = 2

            logger.info("✅ ISIC imaging metadata processed: %s", df.shape)
            return df

        except Exception as exc:
            logger.error("Failed to process ISIC imaging metadata: %s", exc)
            return pd.DataFrame()

    def _load_diabetes_lookup(self, mapping_path: Path) -> Dict[str, Dict[int, str]]:
        """Parse the IDS mapping CSV into lookup dictionaries."""
        if not mapping_path.exists():
            logger.warning(f"Diabetes ID mapping not found at {mapping_path}")
            return {}

        try:
            sections: List[List[str]] = []
            current: List[str] = []

            with mapping_path.open("r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    if line == ",":
                        if current:
                            sections.append(current)
                            current = []
                        continue
                    current.append(line)

            if current:
                sections.append(current)

            lookup: Dict[str, Dict[int, str]] = {}

            for section in sections:
                header_parts = [part.strip() for part in section[0].split(",")]
                if len(header_parts) != 2:
                    continue
                column_name = header_parts[0]
                mapping: Dict[int, str] = {}

                for row in section[1:]:
                    parts = row.split(",", 1)
                    if len(parts) != 2:
                        continue
                    code_raw, desc_raw = parts
                    code_raw = code_raw.strip().strip('"')
                    desc = desc_raw.strip().strip('"')

                    if not desc or desc.upper() == "NULL":
                        desc = "Unknown"

                    if not code_raw or code_raw.upper() == "NULL":
                        continue

                    try:
                        code = int(float(code_raw))
                    except ValueError:
                        logger.debug(
                            "Skipping unmapped ID '%s' in mapping for %s",
                            code_raw,
                            column_name,
                        )
                        continue

                    mapping[code] = desc

                if mapping:
                    lookup[column_name] = mapping

            return lookup

        except Exception as exc:
            logger.error(f"Failed to parse diabetes ID mapping: {exc}")
            return {}

    def process_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Process all available real datasets."""
        logger.info("🔄 Processing all real healthcare datasets...")

        datasets = {
            "heart_disease_uci": self.process_heart_disease_uci(),
            "heart_disease_multicenter": self.process_heart_disease_multicenter(),
            "breast_cancer_wisconsin": self.process_breast_cancer_wisconsin(),
            "diabetes_130_hospitals": self.process_diabetes_130_hospitals(),
            "hepatitis": self.process_hepatitis_dataset(),
            "dermatology": self.process_dermatology_dataset(),
            "medical_appointments": self.process_medical_appointments(),
            "covid_symptoms": self.process_covid_symptoms(),
            "thyroid_disease": self.process_thyroid_disease(),
            "synthea_longitudinal": self.process_synthea_longitudinal(),
            "medical_question_pairs": self.process_medical_question_pairs(),
            "drug_reviews": self.process_drug_reviews(),
            "skin_cancer_imaging": self.process_skin_cancer_imaging_metadata(),
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
                "Dataset '%s' not found. Available: %s",
                dataset_name,
                list(self.processed_datasets.keys()),
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


def main() -> Dict[str, pd.DataFrame]:
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

    print("\n🎉 All datasets saved to: data/processed/")
    print("📋 Summary available at: data/processed/dataset_summary.yaml")

    return datasets


if __name__ == "__main__":
    main()
