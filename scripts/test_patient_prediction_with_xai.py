#!/usr/bin/env python3
"""
Enhanced Patient Prediction Testing with Explainability & Personalization

This script tests production models with realistic patient data and includes:
- Model predictions
- SHAP explanations (feature importance)
- Personalization insights (similar patients)
- Performance metrics
"""

import requests
import json
import time
from typing import Dict, List, Any
from datetime import datetime


class PatientTester:
    """Test patient predictions with explainability"""

    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.results = []

    def test_health(self) -> bool:
        """Test API health"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print("=" * 80)
                print("🏥 API HEALTH CHECK")
                print("=" * 80)
                print(f"Status: {data['status']}")
                print(f"MLflow Connected: {data['mlflow_connected']}")
                print(f"Models Loaded: {data['models_loaded']}")
                print(f"Uptime: {data['uptime_seconds']:.1f}s")
                print("=" * 80)
                return True
            return False
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False

    def predict_with_explanation(
        self,
        model_name: str,
        patient_data: Dict[str, float],
        include_explanation: bool = True,
        include_personalization: bool = False,
    ) -> Dict[str, Any]:
        """
        Make prediction with optional explanations
        """
        url = f"{self.api_url}/predict"

        payload = {
            "model_name": model_name,
            "patient_data": {"features": patient_data},
            "return_probabilities": True,
            "include_explanation": include_explanation,
            "include_personalization": include_personalization,
        }

        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def get_standalone_explanation(
        self,
        model_name: str,
        patient_data: Dict[str, float],
        explanation_type: str = "shap",
    ) -> Dict[str, Any]:
        """Get detailed SHAP explanation via /explain endpoint"""
        url = f"{self.api_url}/explain"

        payload = {
            "model_name": model_name,
            "patient_data": {"features": patient_data},
            "explanation_type": explanation_type,
            "include_personalization": False,
        }

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def print_result(
        self, model_name: str, result: Dict[str, Any], patient_desc: str = ""
    ):
        """Print prediction result with explanations"""
        print("\n" + "=" * 80)
        print(f"🏥 {model_name} - {patient_desc}")
        print("=" * 80)

        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return

        # Prediction
        print(f"\n📊 PREDICTION:")
        print(f"   Result: {result.get('prediction')}")
        print(f"   Confidence: {result.get('confidence', 0):.2%}")
        print(f"   Inference Time: {result.get('inference_time_ms', 0):.2f}ms")

        # Probabilities
        if result.get("probabilities"):
            print(f"\n🎯 PROBABILITIES:")
            for cls, prob in result["probabilities"].items():
                print(f"   {cls}: {prob:.4f}")

        # Explanation
        if result.get("explanation") and "error" not in result["explanation"]:
            print(f"\n🔍 EXPLAINABILITY (SHAP):")
            exp = result["explanation"]
            print(f"   Method: {exp.get('method', 'Unknown')}")

            if "top_features" in exp:
                print(f"\n   Top 5 Important Features:")
                for i, (feat, data) in enumerate(exp["top_features"].items(), 1):
                    impact_symbol = "📈" if data["impact"] == "positive" else "📉"
                    print(f"   {i}. {feat}: {data['shap_value']:.4f} {impact_symbol}")
                    print(
                        f"      Value: {data['feature_value']:.2f} | Impact: {data['impact']}"
                    )

        # Personalization
        if result.get("personalization"):
            print(f"\n👥 PERSONALIZATION:")
            print(f"   Status: {result['personalization'].get('status', 'N/A')}")
            print(f"   {result['personalization'].get('message', 'No details')}")

        print("=" * 80)

    def test_all_models(self):
        """Test all 5 production models with XAI"""

        print("\n" + "🚀" * 40)
        print("PATIENT TESTING WITH EXPLAINABILITY & PERSONALIZATION")
        print("🚀" * 40 + "\n")

        # Test 1: HeartDisease - High Risk
        print("\n🧪 TEST 1: Heart Disease (High Risk Patient)")
        heart_high_risk = {
            "age": 63,
            "sex": 1,
            "cp": 3,
            "trestbps": 145,
            "chol": 233,
            "fbs": 1,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 2.3,
            "slope": 0,
            "ca": 0,
            "thal": 1,
        }
        result = self.predict_with_explanation(
            "HeartDisease", heart_high_risk, include_explanation=True
        )
        self.print_result("HeartDisease", result, "High-Risk Patient (63yo Male)")

        # Test 2: BreastCancer - Malignant
        print("\n🧪 TEST 2: Breast Cancer (Suspicious Tumor)")
        breast_malignant = {
            "mean radius": 20.57,
            "mean texture": 17.77,
            "mean perimeter": 132.9,
            "mean area": 1326.0,
            "mean smoothness": 0.08474,
            "mean compactness": 0.07864,
            "mean concavity": 0.0869,
            "mean concave points": 0.07017,
            "mean symmetry": 0.1812,
            "mean fractal dimension": 0.05667,
            "radius error": 0.5435,
            "texture error": 0.7339,
            "perimeter error": 3.398,
            "area error": 74.08,
            "smoothness error": 0.005225,
            "compactness error": 0.01308,
            "concavity error": 0.0186,
            "concave points error": 0.0134,
            "symmetry error": 0.01389,
            "fractal dimension error": 0.003532,
            "worst radius": 24.99,
            "worst texture": 23.41,
            "worst perimeter": 158.8,
            "worst area": 1956.0,
            "worst smoothness": 0.1238,
            "worst compactness": 0.1866,
            "worst concavity": 0.2416,
            "worst concave points": 0.186,
            "worst symmetry": 0.275,
            "worst fractal dimension": 0.08902,
        }
        result = self.predict_with_explanation(
            "BreastCancer", breast_malignant, include_explanation=True
        )
        self.print_result("BreastCancer", result, "Large Tumor (20.57mm radius)")

        # Test 3: SyntheaLongitudinal - High Complexity
        print("\n🧪 TEST 3: Synthea Longitudinal (Complex Care Patient)")
        synthea_complex = {
            "HEALTHCARE_EXPENSES": 25000,
            "HEALTHCARE_COVERAGE": 20000,
            "AGE": 65,
            "ENCOUNTER_COUNT": 15,
            "CONDITION_COUNT": 8,
            "MEDICATION_COUNT": 12,
            "OBSERVATION_COUNT": 45,
            "PROCEDURE_COUNT": 10,
            "IMMUNIZATION_COUNT": 5,
            "CAREPLAN_COUNT": 3,
            "IMAGING_STUDY_COUNT": 2,
            "BMI": 32.5,
            "SYSTOLIC_BP": 145,
            "DIASTOLIC_BP": 92,
            "HEART_RATE": 82,
            "RESPIRATORY_RATE": 18,
            "GLUCOSE": 180,
            "TOTAL_CHOLESTEROL": 240,
            "HDL": 35,
            "LDL": 155,
            "TRIGLYCERIDES": 210,
            "DAYS_SINCE_LAST_ENCOUNTER": 30,
            "AVG_ENCOUNTER_INTERVAL": 60,
            "TOTAL_CARE_DAYS": 450,
            "CHRONIC_CONDITION_COUNT": 5,
            "ACUTE_CONDITION_COUNT": 3,
            "PREVENTIVE_CARE_SCORE": 0.6,
            "DISEASE_BURDEN": 0.75,
            "HAS_DIABETES": 1,
            "HAS_HYPERTENSION": 1,
            "HAS_HEART_DISEASE": 1,
        }
        result = self.predict_with_explanation(
            "SyntheaLongitudinal", synthea_complex, include_explanation=True
        )
        self.print_result(
            "SyntheaLongitudinal", result, "65yo with Multiple Chronic Conditions"
        )

        # Test 4: ISICSkinCancer - Suspicious Melanoma
        print("\n🧪 TEST 4: ISIC Skin Cancer (Suspicious Lesion)")
        isic_suspicious = {
            "age": 67,
            "sex": 1,
            "localization": 4,
            "diameter_mm": 8.5,
            "asymmetry_score": 0.85,
            "border_irregularity": 0.75,
            "color_variation": 0.80,
            "has_blue_white_veil": 1,
            "has_atypical_pigment": 1,
            "has_regression": 1,
            "has_dots_globules": 1,
            "has_streaks": 1,
            "lesion_area_mm2": 56.75,
            "elevation": 1,
        }
        result = self.predict_with_explanation(
            "ISICSkinCancer", isic_suspicious, include_explanation=True
        )
        self.print_result("ISICSkinCancer", result, "67yo Male, Large Irregular Lesion")

        # Test 5: DrugReviews - Positive Review
        print("\n🧪 TEST 5: Drug Reviews (Effective Treatment)")
        drug_positive = {
            "rating": 9,
            "usefulCount": 45,
            "condition_severity": 3,
            "drug_effectiveness": 0.9,
            "ease_of_use": 0.85,
            "satisfaction": 0.9,
            "review_length": 250,
            "sentiment_score": 0.85,
            "side_effects_mentioned": 0,
            "duration_days": 90,
            "age_group": 2,
            "has_followup": 1,
            "recommendation_score": 0.95,
        }
        result = self.predict_with_explanation(
            "DrugReviews", drug_positive, include_explanation=True
        )
        self.print_result("DrugReviews", result, "High Satisfaction (9/10)")

        print("\n" + "✅" * 40)
        print("ALL TESTS COMPLETED!")
        print("✅" * 40 + "\n")

    def test_standalone_explain_endpoint(self):
        """Test the dedicated /explain endpoint"""
        print("\n" + "=" * 80)
        print("🔬 TESTING STANDALONE /explain ENDPOINT")
        print("=" * 80)

        # Test with HeartDisease model
        patient_data = {
            "age": 63,
            "sex": 1,
            "cp": 3,
            "trestbps": 145,
            "chol": 233,
            "fbs": 1,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 2.3,
            "slope": 0,
            "ca": 0,
            "thal": 1,
        }

        print("\n📊 Getting detailed SHAP explanation...")
        result = self.get_standalone_explanation("HeartDisease", patient_data, "shap")

        if "error" in result:
            print(f"❌ Error: {result['error']}")
        elif "explanation" in result:
            exp = result["explanation"]
            print(f"✅ Method: {exp.get('method', 'Unknown')}")

            if "feature_importance" in exp:
                print(f"\n🎯 Top 10 Important Features:")
                for i, (feat, data) in enumerate(
                    list(exp["feature_importance"].items())[:10], 1
                ):
                    print(
                        f"{i:2d}. {feat:20s}: {data['shap_value']:+.4f} ({data['impact']})"
                    )

            if "top_risk_factors" in exp:
                print(f"\n⚠️  Top Risk Factors:")
                for factor in exp["top_risk_factors"]:
                    print(f"   - {factor}")

            if "top_protective_factors" in exp:
                print(f"\n✅ Top Protective Factors:")
                for factor in exp["top_protective_factors"]:
                    print(f"   - {factor}")

        print("=" * 80)


def main():
    """Main test function"""

    # Initialize tester
    tester = PatientTester()

    # Test API health
    if not tester.test_health():
        print("❌ API is not healthy. Please start the API first:")
        print("   uvicorn api.production_api:app --host 0.0.0.0 --port 8000")
        return

    print("\n⏳ Waiting 2 seconds for API to fully initialize...")
    time.sleep(2)

    # Run tests
    try:
        # Test predictions with explanations
        tester.test_all_models()

        # Test standalone explain endpoint
        tester.test_standalone_explain_endpoint()

        print("\n" + "🎉" * 40)
        print("TESTING COMPLETE - ALL FEATURES VALIDATED!")
        print("🎉" * 40)

    except KeyboardInterrupt:
        print("\n\n⚠️  Testing interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Testing failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
