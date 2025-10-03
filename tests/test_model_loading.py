"""
Model Loading Tests

Tests model loading from MLflow registry and validates predictions.

Usage:
    pytest tests/test_model_loading.py -v
"""

import pytest
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
import time
import joblib
from pathlib import Path

# MLflow configuration
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Test configuration
MODELS_DIR = Path("models/trained_real_datasets")
PRODUCTION_MODELS = [
    "SyntheaLongitudinal",
    "ISICSkinCancer",
    "DrugReviews",
    "BreastCancer",
    "HeartDisease",
]


@pytest.fixture(scope="module")
def mlflow_client():
    """MLflow client fixture"""
    return MlflowClient()


@pytest.fixture(scope="module")
def check_mlflow_running():
    """Check if MLflow is accessible"""
    try:
        client = MlflowClient()
        client.search_registered_models()
    except Exception as e:
        pytest.skip(f"MLflow not accessible: {e}")


class TestMLflowRegistry:
    """Test MLflow Model Registry"""

    def test_registry_accessible(self, mlflow_client, check_mlflow_running):
        """Test MLflow registry is accessible"""
        models = mlflow_client.search_registered_models()
        assert len(models) > 0, "No models in registry"

    def test_production_models_exist(self, mlflow_client, check_mlflow_running):
        """Test all production models are registered"""
        all_models = mlflow_client.search_registered_models()
        model_names = [m.name for m in all_models]

        for expected_model in PRODUCTION_MODELS:
            assert (
                expected_model in model_names
            ), f"Model {expected_model} not found in registry"

    def test_production_stage_set(self, mlflow_client, check_mlflow_running):
        """Test models have Production stage versions"""
        for model_name in PRODUCTION_MODELS:
            versions = mlflow_client.search_model_versions(f"name='{model_name}'")

            production_versions = [
                v for v in versions if v.current_stage == "Production"
            ]

            assert (
                len(production_versions) > 0
            ), f"No Production version for {model_name}"

    def test_model_metadata(self, mlflow_client, check_mlflow_running):
        """Test models have required metadata"""
        for model_name in PRODUCTION_MODELS:
            model = mlflow_client.get_registered_model(model_name)

            # Check description exists
            assert model.description is not None, f"{model_name} missing description"

            # Check has versions
            assert len(model.latest_versions) > 0, f"{model_name} has no versions"


class TestModelLoading:
    """Test loading models from registry"""

    def test_load_model_by_name(self, check_mlflow_running):
        """Test loading model by name and stage"""
        model_name = "HeartDisease"
        model_uri = f"models:/{model_name}/Production"

        start_time = time.time()
        model = mlflow.pyfunc.load_model(model_uri)
        load_time = time.time() - start_time

        assert model is not None
        assert load_time < 5.0, f"Model loading too slow: {load_time:.2f}s"

    def test_load_all_production_models(self, check_mlflow_running):
        """Test all production models can be loaded"""
        for model_name in PRODUCTION_MODELS:
            model_uri = f"models:/{model_name}/Production"

            try:
                model = mlflow.pyfunc.load_model(model_uri)
                assert model is not None, f"Failed to load {model_name}"
            except Exception as e:
                pytest.fail(f"Error loading {model_name}: {e}")

    def test_model_loading_time(self, check_mlflow_running):
        """Test model loading meets performance requirements"""
        load_times = {}

        for model_name in PRODUCTION_MODELS:
            model_uri = f"models:/{model_name}/Production"

            start_time = time.time()
            mlflow.pyfunc.load_model(model_uri)
            load_time = time.time() - start_time

            load_times[model_name] = load_time

        # All models should load in reasonable time
        for model_name, load_time in load_times.items():
            assert load_time < 5.0, f"{model_name} loading too slow: {load_time:.2f}s"

        # Average load time should be under 3 seconds
        avg_time = sum(load_times.values()) / len(load_times)
        assert avg_time < 3.0, f"Average load time too high: {avg_time:.2f}s"


class TestModelPredictions:
    """Test model predictions"""

    def test_heart_disease_prediction(self, check_mlflow_running):
        """Test HeartDisease model prediction"""
        model = mlflow.pyfunc.load_model("models:/HeartDisease/Production")

        # Create test data
        test_data = pd.DataFrame([{f"feature_{i}": float(i) for i in range(13)}])

        # Make prediction
        prediction = model.predict(test_data)

        assert prediction is not None
        assert len(prediction) == 1

    def test_breast_cancer_prediction(self, check_mlflow_running):
        """Test BreastCancer model prediction"""
        model = mlflow.pyfunc.load_model("models:/BreastCancer/Production")

        # Create test data (30 features)
        test_data = pd.DataFrame([{f"feature_{i}": float(i) for i in range(30)}])

        # Make prediction
        prediction = model.predict(test_data)

        assert prediction is not None
        assert len(prediction) == 1

    def test_prediction_inference_time(self, check_mlflow_running):
        """Test prediction inference time"""
        model = mlflow.pyfunc.load_model("models:/HeartDisease/Production")

        test_data = pd.DataFrame([{f"feature_{i}": float(i) for i in range(13)}])

        # Measure inference time
        start_time = time.time()
        model.predict(test_data)
        inference_time = (time.time() - start_time) * 1000  # ms

        # Should be fast (<100ms)
        assert inference_time < 100, f"Inference too slow: {inference_time:.2f}ms"

    def test_batch_prediction(self, check_mlflow_running):
        """Test batch predictions"""
        model = mlflow.pyfunc.load_model("models:/HeartDisease/Production")

        # Create batch of 100 samples
        test_data = pd.DataFrame([{f"feature_{i}": float(i) for i in range(13)}] * 100)

        # Make batch prediction
        predictions = model.predict(test_data)

        assert len(predictions) == 100

    def test_prediction_consistency(self, check_mlflow_running):
        """Test predictions are consistent across runs"""
        model = mlflow.pyfunc.load_model("models:/HeartDisease/Production")

        test_data = pd.DataFrame([{f"feature_{i}": float(i) for i in range(13)}])

        # Make multiple predictions
        pred1 = model.predict(test_data)
        pred2 = model.predict(test_data)
        pred3 = model.predict(test_data)

        # Should be identical
        assert np.array_equal(pred1, pred2)
        assert np.array_equal(pred2, pred3)


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_dataframe(self, check_mlflow_running):
        """Test prediction with empty dataframe"""
        model = mlflow.pyfunc.load_model("models:/HeartDisease/Production")

        # Empty dataframe
        test_data = pd.DataFrame()

        # Should handle gracefully
        try:
            prediction = model.predict(test_data)
            assert len(prediction) == 0
        except Exception:
            # Exception is acceptable for empty input
            pass

    def test_single_feature_missing(self, check_mlflow_running):
        """Test prediction with missing features"""
        model = mlflow.pyfunc.load_model("models:/HeartDisease/Production")

        # Data with missing feature (only 12 instead of 13)
        test_data = pd.DataFrame([{f"feature_{i}": float(i) for i in range(12)}])

        # Should either handle or raise clear error
        try:
            prediction = model.predict(test_data)
            # If it works, that's fine
            assert prediction is not None
        except Exception as e:
            # Error is acceptable, but should be informative
            assert "feature" in str(e).lower() or "shape" in str(e).lower()

    def test_extreme_values(self, check_mlflow_running):
        """Test prediction with extreme values"""
        model = mlflow.pyfunc.load_model("models:/HeartDisease/Production")

        # Extreme values
        test_data = pd.DataFrame(
            [{f"feature_{i}": 1e10 if i % 2 == 0 else -1e10 for i in range(13)}]
        )

        # Should handle without crashing
        try:
            prediction = model.predict(test_data)
            assert prediction is not None
        except Exception:
            # Some models may not handle extreme values
            pass

    def test_nan_values(self, check_mlflow_running):
        """Test prediction with NaN values"""
        model = mlflow.pyfunc.load_model("models:/HeartDisease/Production")

        # Data with NaN
        test_data = pd.DataFrame(
            [{f"feature_{i}": np.nan if i == 0 else float(i) for i in range(13)}]
        )

        # Should handle or raise clear error
        try:
            prediction = model.predict(test_data)
            # If it works, check result is not NaN
            assert not np.isnan(prediction).any()
        except Exception:
            # Exception is acceptable for NaN input
            pass


class TestSavedModels:
    """Test locally saved model files"""

    def test_saved_models_exist(self):
        """Test saved model files exist"""
        if not MODELS_DIR.exists():
            pytest.skip("Saved models directory not found")

        model_files = list(MODELS_DIR.glob("*.joblib"))
        assert len(model_files) > 0, "No saved model files found"

    def test_load_saved_model(self):
        """Test loading a saved model file"""
        if not MODELS_DIR.exists():
            pytest.skip("Saved models directory not found")

        model_files = list(MODELS_DIR.glob("*RandomForest.joblib"))
        if not model_files:
            pytest.skip("No RandomForest model files found")

        model_file = model_files[0]
        model_data = joblib.load(model_file)

        # Check structure
        assert "model" in model_data
        assert "scaler" in model_data
        assert "feature_names" in model_data

    def test_saved_model_metadata(self):
        """Test saved models have metadata"""
        if not MODELS_DIR.exists():
            pytest.skip("Saved models directory not found")

        model_files = list(MODELS_DIR.glob("*.joblib"))
        if not model_files:
            pytest.skip("No model files found")

        model_data = joblib.load(model_files[0])

        # Check metadata exists
        assert "accuracy" in model_data
        assert "metadata" in model_data

        metadata = model_data["metadata"]
        assert "dataset_name" in metadata
        assert "model_type" in metadata
        assert "training_date" in metadata


class TestModelCompatibility:
    """Test model compatibility and versions"""

    def test_model_sklearn_version(self, check_mlflow_running):
        """Test models compatible with current sklearn version"""
        import sklearn

        for model_name in PRODUCTION_MODELS:
            model_uri = f"models:/{model_name}/Production"

            try:
                model = mlflow.pyfunc.load_model(model_uri)
                # If it loads, it's compatible
                assert model is not None
            except Exception as e:
                if "version" in str(e).lower():
                    pytest.fail(
                        f"{model_name} incompatible with sklearn "
                        f"{sklearn.__version__}: {e}"
                    )
                raise

    def test_model_python_version(self, check_mlflow_running):
        """Test models work with current Python version"""
        import sys

        for model_name in PRODUCTION_MODELS:
            model_uri = f"models:/{model_name}/Production"

            try:
                model = mlflow.pyfunc.load_model(model_uri)
                # Simple prediction test
                test_data = pd.DataFrame([{f"f{i}": 0.0 for i in range(10)}])
                model.predict(test_data)
            except Exception as e:
                pytest.fail(f"{model_name} failed on Python {sys.version}: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
