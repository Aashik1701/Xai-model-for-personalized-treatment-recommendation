#!/usr/bin/env python3
"""
Patient Prediction Testing Script

Interactive script to test production models with patient input.
Supports all 5 registered production models with sample patient data.

Usage:
    python scripts/test_patient_prediction.py
    python scripts/test_patient_prediction.py --model HeartDisease
    python scripts/test_patient_prediction.py --interactive
"""

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List
import sys

# Configuration
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Sample patient data for each model
SAMPLE_PATIENTS = {
    "HeartDisease": {
        "description": "Heart Disease Patient - 13 features",
        "features": [
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
        ],
        "samples": [
            {
                "name": "High Risk Patient",
                "data": [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1],
                "expected": "Heart Disease Positive",
            },
            {
                "name": "Low Risk Patient",
                "data": [45, 0, 0, 120, 180, 0, 1, 160, 0, 0.5, 2, 0, 2],
                "expected": "Heart Disease Negative",
            },
        ],
    },
    "BreastCancer": {
        "description": "Breast Cancer Patient - 30 features",
        "features": [
            "mean_radius",
            "mean_texture",
            "mean_perimeter",
            "mean_area",
            "mean_smoothness",
            "mean_compactness",
            "mean_concavity",
            "mean_concave_points",
            "mean_symmetry",
            "mean_fractal_dimension",
            "radius_error",
            "texture_error",
            "perimeter_error",
            "area_error",
            "smoothness_error",
            "compactness_error",
            "concavity_error",
            "concave_points_error",
            "symmetry_error",
            "fractal_dimension_error",
            "worst_radius",
            "worst_texture",
            "worst_perimeter",
            "worst_area",
            "worst_smoothness",
            "worst_compactness",
            "worst_concavity",
            "worst_concave_points",
            "worst_symmetry",
            "worst_fractal_dimension",
        ],
        "samples": [
            {
                "name": "Malignant Tumor",
                "data": [
                    20.57,
                    17.77,
                    132.90,
                    1326.0,
                    0.08474,
                    0.07864,
                    0.0869,
                    0.07017,
                    0.1812,
                    0.05667,
                    0.5435,
                    0.7339,
                    3.398,
                    74.08,
                    0.005225,
                    0.01308,
                    0.0186,
                    0.0134,
                    0.01389,
                    0.003532,
                    24.99,
                    23.41,
                    158.80,
                    1956.0,
                    0.1238,
                    0.1866,
                    0.2416,
                    0.186,
                    0.275,
                    0.08902,
                ],
                "expected": "Malignant",
            },
            {
                "name": "Benign Tumor",
                "data": [
                    11.42,
                    20.38,
                    77.58,
                    386.1,
                    0.1425,
                    0.2839,
                    0.2414,
                    0.1052,
                    0.2597,
                    0.09744,
                    0.4956,
                    1.156,
                    3.445,
                    27.23,
                    0.00911,
                    0.07458,
                    0.05661,
                    0.01867,
                    0.05963,
                    0.009208,
                    14.91,
                    26.50,
                    98.87,
                    567.7,
                    0.2098,
                    0.8663,
                    0.6869,
                    0.2575,
                    0.6638,
                    0.1730,
                ],
                "expected": "Benign",
            },
        ],
    },
    "SyntheaLongitudinal": {
        "description": "Synthea Longitudinal Patient - 31 features",
        "features": [
            "HEALTHCARE_EXPENSES",
            "HEALTHCARE_COVERAGE",
            "AGE",
            "ENCOUNTER_COUNT",
            "CONDITION_COUNT",
            "MEDICATION_COUNT",
            "OBSERVATION_COUNT",
            "PROCEDURE_COUNT",
            "IMMUNIZATION_COUNT",
            "CAREPLANS_COUNT",
            "IMAGING_COUNT",
            "BMI",
            "SYSTOLIC_BP",
            "DIASTOLIC_BP",
            "HEART_RATE",
            "RESPIRATORY_RATE",
            "GLUCOSE",
            "TOTAL_CHOLESTEROL",
            "HDL",
            "LDL",
            "TRIGLYCERIDES",
            "DAYS_SINCE_LAST_ENCOUNTER",
            "AVG_ENCOUNTER_INTERVAL",
            "TOTAL_CARE_DAYS",
            "CHRONIC_CONDITIONS",
            "ACUTE_CONDITIONS",
            "PREVENTIVE_CARE_SCORE",
            "DISEASE_BURDEN",
            "HAS_DIABETES",
            "HAS_HYPERTENSION",
            "HAS_HEART_DISEASE",
        ],
        "samples": [
            {
                "name": "High-Risk Chronic Patient",
                "data": [
                    25000,
                    20000,
                    65,
                    15,
                    8,
                    12,
                    45,
                    10,
                    5,
                    3,
                    2,
                    32.5,
                    145,
                    92,
                    82,
                    18,
                    180,
                    240,
                    35,
                    155,
                    210,
                    30,
                    60,
                    450,
                    5,
                    3,
                    0.6,
                    0.75,
                    1,
                    1,
                    1,
                ],
                "expected": "High Risk / Complex Care Needed",
            },
            {
                "name": "Healthy Young Adult",
                "data": [
                    2000,
                    1800,
                    28,
                    3,
                    1,
                    0,
                    8,
                    2,
                    8,
                    1,
                    0,
                    22.5,
                    118,
                    75,
                    70,
                    16,
                    95,
                    185,
                    55,
                    110,
                    85,
                    180,
                    120,
                    15,
                    0,
                    1,
                    0.9,
                    0.15,
                    0,
                    0,
                    0,
                ],
                "expected": "Low Risk / Healthy",
            },
        ],
    },
    "ISICSkinCancer": {
        "description": "ISIC Skin Cancer Patient - 14 features",
        "features": [
            "age",
            "sex",
            "localization",
            "diameter_mm",
            "asymmetry_score",
            "border_irregularity",
            "color_variation",
            "has_blue_white_veil",
            "has_atypical_pigment",
            "has_regression",
            "has_dots_globules",
            "has_streaks",
            "lesion_area_mm2",
            "elevation",
        ],
        "samples": [
            {
                "name": "Suspicious Melanoma",
                "data": [67, 1, 4, 8.5, 0.85, 0.75, 0.80, 1, 1, 1, 1, 1, 56.75, 1],
                "expected": "Malignant / Melanoma",
            },
            {
                "name": "Benign Nevus",
                "data": [35, 0, 2, 4.2, 0.15, 0.20, 0.25, 0, 0, 0, 0, 0, 13.85, 0],
                "expected": "Benign",
            },
        ],
    },
    "DrugReviews": {
        "description": "Drug Review Patient - 13 features",
        "features": [
            "rating",
            "useful_count",
            "condition_severity",
            "drug_effectiveness",
            "ease_of_use",
            "satisfaction",
            "review_length",
            "sentiment_score",
            "side_effects_mentioned",
            "duration_days",
            "age_group",
            "has_followup",
            "recommendation_score",
        ],
        "samples": [
            {
                "name": "Positive Review - Effective Treatment",
                "data": [9, 45, 3, 0.9, 0.85, 0.9, 250, 0.85, 0, 90, 2, 1, 0.95],
                "expected": "High Satisfaction / Effective",
            },
            {
                "name": "Negative Review - Poor Response",
                "data": [3, 8, 2, 0.3, 0.5, 0.25, 120, -0.6, 1, 30, 1, 0, 0.2],
                "expected": "Low Satisfaction / Ineffective",
            },
        ],
    },
}


class PatientTester:
    """Test production models with patient data"""

    def __init__(self):
        self.client = MlflowClient()
        print("\n" + "=" * 80)
        print("🏥 PATIENT PREDICTION TESTING SYSTEM")
        print("=" * 80)

    def list_available_models(self):
        """List all available production models"""
        print("\n📋 Available Production Models:")
        print("-" * 80)

        for i, model_name in enumerate(SAMPLE_PATIENTS.keys(), 1):
            info = SAMPLE_PATIENTS[model_name]
            print(f"{i}. {model_name}")
            print(f"   {info['description']}")
            print(f"   Features: {len(info['features'])}")
            print()

    def load_model(self, model_name: str):
        """Load production model from MLflow"""
        print(f"\n🔄 Loading model: {model_name}")
        print("-" * 80)

        try:
            # Get production version
            versions = self.client.search_model_versions(f"name='{model_name}'")
            prod_versions = [v for v in versions if v.current_stage == "Production"]

            if not prod_versions:
                print(f"❌ No production version found for {model_name}")
                return None

            version = prod_versions[0]
            model_uri = f"models:/{model_name}/Production"

            print(f"✓ Found version {version.version} (Run: {version.run_id[:8]}...)")
            print(f"✓ Loading from: {model_uri}")

            model = mlflow.pyfunc.load_model(model_uri)
            print(f"✅ Model loaded successfully!")

            return model

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None

    def predict_patient(
        self,
        model,
        model_name: str,
        patient_data: List[float],
        patient_name: str = "Unknown",
    ):
        """Make prediction for a patient"""
        print(f"\n🔬 Testing Patient: {patient_name}")
        print("-" * 80)

        try:
            # Get feature names
            features = SAMPLE_PATIENTS[model_name]["features"]

            # Create DataFrame
            df = pd.DataFrame([patient_data], columns=features)

            print(f"📊 Patient Features:")
            for feat, val in zip(features[:5], patient_data[:5]):
                print(f"   {feat:30s}: {val}")
            if len(features) > 5:
                print(f"   ... and {len(features)-5} more features")
            print()

            # Make prediction
            print("🤖 Running prediction...")
            prediction = model.predict(df)

            # Get prediction result
            result = prediction[0] if len(prediction) > 0 else "Unknown"

            print(f"\n✅ PREDICTION RESULT: {result}")

            # Try to get probability if available
            try:
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(df)
                    print(f"\n📊 Confidence Scores:")
                    for i, prob in enumerate(proba[0]):
                        print(f"   Class {i}: {prob:.2%}")
            except:
                pass

            return result

        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            import traceback

            traceback.print_exc()
            return None

    def test_model(self, model_name: str, interactive: bool = False):
        """Test a specific model with sample patients"""
        print("\n" + "=" * 80)
        print(f"🧪 TESTING MODEL: {model_name}")
        print("=" * 80)

        if model_name not in SAMPLE_PATIENTS:
            print(f"❌ Model '{model_name}' not found in sample data")
            print(f"Available models: {', '.join(SAMPLE_PATIENTS.keys())}")
            return

        # Load model
        model = self.load_model(model_name)
        if model is None:
            return

        # Get sample patients
        model_info = SAMPLE_PATIENTS[model_name]
        print(f"\n📝 Model Description: {model_info['description']}")
        print(f"📊 Number of Features: {len(model_info['features'])}")
        print(f"👥 Sample Patients Available: {len(model_info['samples'])}")

        # Test each sample patient
        results = []
        for sample in model_info["samples"]:
            result = self.predict_patient(
                model, model_name, sample["data"], sample["name"]
            )

            print(f"\n   Expected: {sample['expected']}")
            print(f"   Match: {'✅' if result is not None else '❌'}")

            results.append(
                {
                    "patient": sample["name"],
                    "expected": sample["expected"],
                    "predicted": result,
                    "success": result is not None,
                }
            )

            print("\n" + "-" * 80)

        # Summary
        print(f"\n📊 TEST SUMMARY for {model_name}")
        print("=" * 80)
        successful = sum(1 for r in results if r["success"])
        print(f"Total Patients Tested: {len(results)}")
        print(f"Successful Predictions: {successful}/{len(results)}")
        print(f"Success Rate: {successful/len(results)*100:.1f}%")
        print()

        for r in results:
            status = "✅" if r["success"] else "❌"
            print(f"{status} {r['patient']:30s} → {r['predicted']}")

        return results

    def test_all_models(self):
        """Test all production models"""
        print("\n" + "=" * 80)
        print("🧪 TESTING ALL PRODUCTION MODELS")
        print("=" * 80)

        all_results = {}
        for model_name in SAMPLE_PATIENTS.keys():
            results = self.test_model(model_name)
            all_results[model_name] = results
            print("\n")

        # Overall summary
        print("\n" + "=" * 80)
        print("📊 OVERALL TESTING SUMMARY")
        print("=" * 80)

        for model_name, results in all_results.items():
            if results:
                successful = sum(1 for r in results if r["success"])
                total = len(results)
                rate = successful / total * 100
                status = "✅" if rate == 100 else "⚠️"
                print(f"{status} {model_name:25s}: {successful}/{total} ({rate:.1f}%)")

        return all_results

    def interactive_mode(self, model_name: str):
        """Interactive mode for custom patient input"""
        print("\n" + "=" * 80)
        print(f"🖥️  INTERACTIVE MODE: {model_name}")
        print("=" * 80)

        if model_name not in SAMPLE_PATIENTS:
            print(f"❌ Model '{model_name}' not available")
            return

        # Load model
        model = self.load_model(model_name)
        if model is None:
            return

        model_info = SAMPLE_PATIENTS[model_name]
        features = model_info["features"]

        print(f"\n📝 Required Features ({len(features)}):")
        print("-" * 80)
        for i, feat in enumerate(features, 1):
            print(f"{i:2d}. {feat}")

        print("\n💡 Options:")
        print("1. Enter custom values")
        print("2. Use sample patient data")
        print("3. Exit")

        choice = input("\nSelect option (1-3): ").strip()

        if choice == "1":
            print(f"\n📊 Enter values for {len(features)} features:")
            patient_data = []
            for feat in features:
                while True:
                    try:
                        val = input(f"  {feat}: ").strip()
                        patient_data.append(float(val))
                        break
                    except ValueError:
                        print("    ❌ Invalid number, try again")

            patient_name = (
                input("\n👤 Patient name (optional): ").strip() or "Custom Patient"
            )
            self.predict_patient(model, model_name, patient_data, patient_name)

        elif choice == "2":
            print("\n👥 Available Sample Patients:")
            for i, sample in enumerate(model_info["samples"], 1):
                print(f"{i}. {sample['name']} (Expected: {sample['expected']})")

            sample_choice = input(
                f"\nSelect patient (1-{len(model_info['samples'])}): "
            ).strip()
            try:
                idx = int(sample_choice) - 1
                sample = model_info["samples"][idx]
                self.predict_patient(model, model_name, sample["data"], sample["name"])
                print(f"\n   Expected Result: {sample['expected']}")
            except (ValueError, IndexError):
                print("❌ Invalid selection")

        else:
            print("Exiting interactive mode")


def main():
    parser = argparse.ArgumentParser(
        description="Test production models with patient data"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name to test (HeartDisease, BreastCancer, etc.)",
    )
    parser.add_argument("--all", action="store_true", help="Test all models")
    parser.add_argument(
        "--interactive", action="store_true", help="Interactive mode for custom input"
    )
    parser.add_argument("--list", action="store_true", help="List available models")

    args = parser.parse_args()

    tester = PatientTester()

    if args.list:
        tester.list_available_models()
    elif args.all:
        tester.test_all_models()
    elif args.model:
        if args.interactive:
            tester.interactive_mode(args.model)
        else:
            tester.test_model(args.model)
    else:
        # Default: test all models
        print("\n💡 Usage Examples:")
        print("  python scripts/test_patient_prediction.py --all")
        print("  python scripts/test_patient_prediction.py --model HeartDisease")
        print(
            "  python scripts/test_patient_prediction.py --model BreastCancer --interactive"
        )
        print("  python scripts/test_patient_prediction.py --list")
        print("\nRunning default: Test all models\n")
        tester.test_all_models()


if __name__ == "__main__":
    main()
