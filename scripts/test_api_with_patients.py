#!/usr/bin/env python3
"""
Test Production API with Patient Data

Tests the FastAPI production endpoint with real patient data.
"""

import requests
import json

API_URL = "http://localhost:8000"

# Test data for different models
TEST_PATIENTS = {
    "HeartDisease": {
        "features": {
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
        },
        "description": "High-risk patient with multiple risk factors",
    },
    "BreastCancer": {
        "features": {
            "mean_radius": 20.57,
            "mean_texture": 17.77,
            "mean_perimeter": 132.90,
            "mean_area": 1326.0,
            "mean_smoothness": 0.08474,
            "mean_compactness": 0.07864,
            "mean_concavity": 0.0869,
            "mean_concave_points": 0.07017,
            "mean_symmetry": 0.1812,
            "mean_fractal_dimension": 0.05667,
            "radius_error": 0.5435,
            "texture_error": 0.7339,
            "perimeter_error": 3.398,
            "area_error": 74.08,
            "smoothness_error": 0.005225,
            "compactness_error": 0.01308,
            "concavity_error": 0.0186,
            "concave_points_error": 0.0134,
            "symmetry_error": 0.01389,
            "fractal_dimension_error": 0.003532,
            "worst_radius": 24.99,
            "worst_texture": 23.41,
            "worst_perimeter": 158.80,
            "worst_area": 1956.0,
            "worst_smoothness": 0.1238,
            "worst_compactness": 0.1866,
            "worst_concavity": 0.2416,
            "worst_concave_points": 0.186,
            "worst_symmetry": 0.275,
            "worst_fractal_dimension": 0.08902,
        },
        "description": "Suspected malignant tumor",
    },
}


def test_health():
    """Test health endpoint"""
    print("\n" + "=" * 80)
    print("🏥 TESTING API HEALTH ENDPOINT")
    print("=" * 80)

    try:
        response = requests.get(f"{API_URL}/health")
        print(f"\nStatus Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Status: {data.get('status', 'unknown')}")
            print(f"✅ MLflow Connected: {data.get('mlflow_connected', False)}")
            print(f"✅ Models Available: {data.get('models_count', 0)}")
        else:
            print(f"❌ Health check failed")

    except Exception as e:
        print(f"❌ Error: {e}")


def test_list_models():
    """Test list models endpoint"""
    print("\n" + "=" * 80)
    print("📋 TESTING LIST MODELS ENDPOINT")
    print("=" * 80)

    try:
        response = requests.get(f"{API_URL}/models")
        print(f"\nStatus Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            print(f"\n✅ Found {len(models)} production models:")
            for model in models:
                print(f"   • {model['name']} (v{model['version']})")
        else:
            print(f"❌ Failed to list models")

    except Exception as e:
        print(f"❌ Error: {e}")


def test_prediction(model_name: str, patient_data: dict, description: str):
    """Test prediction endpoint"""
    print("\n" + "=" * 80)
    print(f"🔬 TESTING PREDICTION: {model_name}")
    print("=" * 80)

    print(f"\n👤 Patient: {description}")
    print(f"📊 Features: {len(patient_data)} values")

    try:
        payload = {"model_name": model_name, "features": patient_data}

        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"},
        )

        print(f"\nStatus Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ PREDICTION RESULT")
            print(f"   Model: {data.get('model_name')}")
            print(f"   Version: {data.get('model_version')}")
            print(f"   Prediction: {data.get('prediction')}")

            if "probability" in data:
                print(f"   Confidence: {data['probability']:.2%}")

            print(f"   Timestamp: {data.get('timestamp')}")

        else:
            print(f"❌ Prediction failed")
            print(f"Response: {response.text}")

    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    print("\n" + "=" * 80)
    print("🧪 PRODUCTION API PATIENT TESTING")
    print("=" * 80)
    print(f"\nAPI URL: {API_URL}")

    # Test 1: Health check
    test_health()

    # Test 2: List models
    test_list_models()

    # Test 3: Predictions
    for model_name, test_data in TEST_PATIENTS.items():
        test_prediction(model_name, test_data["features"], test_data["description"])

    print("\n" + "=" * 80)
    print("✅ API TESTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
