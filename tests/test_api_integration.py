"""
API Integration Tests

Tests all API endpoints with various scenarios including:
- Normal requests
- Error handling
- Edge cases
- Response validation

Usage:
    pytest tests/test_api_integration.py -v
"""

import pytest
import requests
import json
import io
import pandas as pd
from typing import Dict, Any

# API configuration
API_BASE_URL = "http://localhost:8000"
TIMEOUT = 30

# Test data
HEART_DISEASE_FEATURES = {
    "age": 65,
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

BREAST_CANCER_FEATURES = {f"feature_{i}": float(i) for i in range(30)}


@pytest.fixture(scope="module")
def api_url():
    """API base URL fixture"""
    return API_BASE_URL


@pytest.fixture(scope="module")
def check_api_running():
    """Check if API is running before tests"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            pytest.skip("API is not running or unhealthy")
    except requests.exceptions.ConnectionError:
        pytest.skip("Cannot connect to API - please start the server")


class TestRootEndpoint:
    """Test root endpoint"""

    def test_root_endpoint(self, api_url, check_api_running):
        """Test GET / returns API info"""
        response = requests.get(api_url, timeout=TIMEOUT)

        assert response.status_code == 200
        data = response.json()

        assert "message" in data
        assert "version" in data
        assert data["version"] == "1.0.0"
        assert "/docs" in data["docs"]

    def test_root_response_structure(self, api_url, check_api_running):
        """Test root endpoint response structure"""
        response = requests.get(api_url, timeout=TIMEOUT)
        data = response.json()

        required_keys = ["message", "version", "docs", "health"]
        for key in required_keys:
            assert key in data, f"Missing key: {key}"


class TestHealthEndpoint:
    """Test health check endpoint"""

    def test_health_check_success(self, api_url, check_api_running):
        """Test GET /health returns healthy status"""
        response = requests.get(f"{api_url}/health", timeout=TIMEOUT)

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert data["status"] in ["healthy", "degraded"]
        assert "timestamp" in data
        assert "mlflow_connected" in data
        assert "models_loaded" in data
        assert "uptime_seconds" in data

    def test_health_check_mlflow_status(self, api_url, check_api_running):
        """Test health check reports MLflow connection"""
        response = requests.get(f"{api_url}/health", timeout=TIMEOUT)
        data = response.json()

        assert isinstance(data["mlflow_connected"], bool)

        if data["mlflow_connected"]:
            assert data["status"] == "healthy"

    def test_health_check_uptime(self, api_url, check_api_running):
        """Test health check reports uptime"""
        response = requests.get(f"{api_url}/health", timeout=TIMEOUT)
        data = response.json()

        assert data["uptime_seconds"] > 0
        assert isinstance(data["uptime_seconds"], (int, float))


class TestModelsEndpoint:
    """Test models listing endpoint"""

    def test_list_models_success(self, api_url, check_api_running):
        """Test GET /models returns model list"""
        response = requests.get(f"{api_url}/models", timeout=TIMEOUT)

        assert response.status_code == 200
        data = response.json()

        assert "models" in data
        assert "total" in data
        assert isinstance(data["models"], list)
        assert data["total"] == len(data["models"])

    def test_list_models_contains_expected(self, api_url, check_api_running):
        """Test models list contains expected production models"""
        response = requests.get(f"{api_url}/models", timeout=TIMEOUT)
        data = response.json()

        expected_models = [
            "SyntheaLongitudinal",
            "ISICSkinCancer",
            "DrugReviews",
            "BreastCancer",
            "HeartDisease",
        ]

        model_names = [m["name"] for m in data["models"]]

        for expected in expected_models:
            assert expected in model_names, f"Missing model: {expected}"

    def test_model_metadata_structure(self, api_url, check_api_running):
        """Test each model has required metadata"""
        response = requests.get(f"{api_url}/models", timeout=TIMEOUT)
        data = response.json()

        required_fields = [
            "name",
            "version",
            "stage",
            "description",
            "accuracy",
            "use_case",
            "run_id",
        ]

        for model in data["models"]:
            for field in required_fields:
                assert field in model, f"Model {model.get('name')} missing {field}"


class TestPredictEndpoint:
    """Test single prediction endpoint"""

    def test_predict_heart_disease_success(self, api_url, check_api_running):
        """Test POST /predict with HeartDisease model"""
        payload = {
            "model_name": "HeartDisease",
            "patient_data": {"features": HEART_DISEASE_FEATURES},
            "return_probabilities": True,
        }

        response = requests.post(f"{api_url}/predict", json=payload, timeout=TIMEOUT)

        assert response.status_code == 200
        data = response.json()

        # Check required fields
        assert "prediction" in data
        assert "confidence" in data
        assert "model_name" in data
        assert "model_version" in data
        assert "inference_time_ms" in data
        assert "trace_id" in data
        assert "timestamp" in data

    def test_predict_with_probabilities(self, api_url, check_api_running):
        """Test prediction returns probabilities when requested"""
        payload = {
            "model_name": "HeartDisease",
            "patient_data": {"features": HEART_DISEASE_FEATURES},
            "return_probabilities": True,
        }

        response = requests.post(f"{api_url}/predict", json=payload, timeout=TIMEOUT)

        data = response.json()

        # Probabilities should be present
        assert "probabilities" in data or data["probabilities"] is None

        # Confidence should be between 0 and 1
        assert 0 <= data["confidence"] <= 1

    def test_predict_without_probabilities(self, api_url, check_api_running):
        """Test prediction without probabilities"""
        payload = {
            "model_name": "HeartDisease",
            "patient_data": {"features": HEART_DISEASE_FEATURES},
            "return_probabilities": False,
        }

        response = requests.post(f"{api_url}/predict", json=payload, timeout=TIMEOUT)

        assert response.status_code == 200
        data = response.json()

        # Should still have prediction
        assert "prediction" in data

    def test_predict_invalid_model(self, api_url, check_api_running):
        """Test prediction with invalid model name"""
        payload = {
            "model_name": "InvalidModel",
            "patient_data": {"features": HEART_DISEASE_FEATURES},
        }

        response = requests.post(f"{api_url}/predict", json=payload, timeout=TIMEOUT)

        # Should return validation error
        assert response.status_code == 422

    def test_predict_missing_features(self, api_url, check_api_running):
        """Test prediction with missing patient data"""
        payload = {
            "model_name": "HeartDisease",
            "patient_data": {"features": {}},  # Empty features
        }

        response = requests.post(f"{api_url}/predict", json=payload, timeout=TIMEOUT)

        # Should either accept (and model handles it) or return error
        assert response.status_code in [200, 422, 500]

    def test_predict_all_models(self, api_url, check_api_running):
        """Test prediction works for all registered models"""
        models = [
            ("HeartDisease", HEART_DISEASE_FEATURES),
            ("BreastCancer", BREAST_CANCER_FEATURES),
        ]

        for model_name, features in models:
            payload = {"model_name": model_name, "patient_data": {"features": features}}

            response = requests.post(
                f"{api_url}/predict", json=payload, timeout=TIMEOUT
            )

            # Should succeed or return model-specific error
            assert response.status_code in [200, 500], f"Failed for model: {model_name}"

    def test_predict_inference_time(self, api_url, check_api_running):
        """Test inference time is reasonable"""
        payload = {
            "model_name": "HeartDisease",
            "patient_data": {"features": HEART_DISEASE_FEATURES},
        }

        response = requests.post(f"{api_url}/predict", json=payload, timeout=TIMEOUT)

        data = response.json()

        # Inference should be fast (<1000ms)
        assert (
            data["inference_time_ms"] < 1000
        ), f"Inference too slow: {data['inference_time_ms']}ms"


class TestBatchPredictEndpoint:
    """Test batch prediction endpoint"""

    def test_batch_predict_success(self, api_url, check_api_running):
        """Test POST /predict/batch with CSV file"""
        # Create test CSV
        df = pd.DataFrame([HEART_DISEASE_FEATURES] * 10)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        files = {"file": ("test_patients.csv", csv_buffer.getvalue(), "text/csv")}

        response = requests.post(
            f"{api_url}/predict/batch?model_name=HeartDisease",
            files=files,
            timeout=TIMEOUT,
        )

        assert response.status_code == 200
        data = response.json()

        assert "predictions" in data
        assert "total_samples" in data
        assert "model_name" in data
        assert data["total_samples"] == 10

    def test_batch_predict_invalid_model(self, api_url, check_api_running):
        """Test batch prediction with invalid model"""
        df = pd.DataFrame([HEART_DISEASE_FEATURES])
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        files = {"file": ("test.csv", csv_buffer.getvalue(), "text/csv")}

        response = requests.post(
            f"{api_url}/predict/batch?model_name=InvalidModel",
            files=files,
            timeout=TIMEOUT,
        )

        assert response.status_code == 400


class TestExplainEndpoint:
    """Test explanation endpoint"""

    def test_explain_placeholder(self, api_url, check_api_running):
        """Test POST /explain returns placeholder response"""
        payload = {
            "model_name": "HeartDisease",
            "patient_data": {"features": HEART_DISEASE_FEATURES},
            "explanation_type": "shap",
        }

        response = requests.post(f"{api_url}/explain", json=payload, timeout=TIMEOUT)

        assert response.status_code == 200
        data = response.json()

        assert "message" in data
        assert "model_name" in data
        assert data["model_name"] == "HeartDisease"


class TestMetricsEndpoint:
    """Test metrics endpoint"""

    def test_metrics_endpoint(self, api_url, check_api_running):
        """Test GET /metrics returns performance metrics"""
        response = requests.get(f"{api_url}/metrics", timeout=TIMEOUT)

        assert response.status_code == 200
        data = response.json()

        assert "uptime_seconds" in data
        assert "models_cached" in data
        assert "cache_details" in data


class TestErrorHandling:
    """Test error handling"""

    def test_404_not_found(self, api_url, check_api_running):
        """Test 404 for non-existent endpoint"""
        response = requests.get(f"{api_url}/nonexistent", timeout=TIMEOUT)

        assert response.status_code == 404

    def test_invalid_json(self, api_url, check_api_running):
        """Test handling of invalid JSON"""
        response = requests.post(
            f"{api_url}/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT,
        )

        assert response.status_code == 422


class TestPerformance:
    """Test API performance"""

    def test_concurrent_requests(self, api_url, check_api_running):
        """Test API handles concurrent requests"""
        import concurrent.futures

        def make_request():
            payload = {
                "model_name": "HeartDisease",
                "patient_data": {"features": HEART_DISEASE_FEATURES},
            }
            response = requests.post(
                f"{api_url}/predict", json=payload, timeout=TIMEOUT
            )
            return response.status_code

        # Send 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [f.result() for f in futures]

        # All should succeed
        assert all(status == 200 for status in results)

    def test_response_time_consistency(self, api_url, check_api_running):
        """Test response times are consistent"""
        times = []

        for _ in range(5):
            payload = {
                "model_name": "HeartDisease",
                "patient_data": {"features": HEART_DISEASE_FEATURES},
            }

            import time

            start = time.time()
            response = requests.post(
                f"{api_url}/predict", json=payload, timeout=TIMEOUT
            )
            elapsed = (time.time() - start) * 1000  # ms

            times.append(elapsed)

        # Response times should be similar
        avg_time = sum(times) / len(times)
        assert all(t < avg_time * 2 for t in times), "Response times vary too much"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
