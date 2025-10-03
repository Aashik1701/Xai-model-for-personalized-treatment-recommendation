"""API tests covering model loading, preprocessing, and SHAP explanations."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Dict, Iterator

import pandas as pd  # type: ignore[import-untyped]
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import make_classification  # type: ignore[import-untyped]


@pytest.fixture(scope="module")
def api_setup(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Dict[str, Any]]:
    tmp_dir = Path(tmp_path_factory.mktemp("pipeline_artifact"))
    X, y = make_classification(
        n_samples=160,
        n_features=6,
        n_informative=4,
        random_state=7,
    )
    feature_cols = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_cols)
    df["target"] = y

    module = importlib.import_module(
        "hybrid_xai_healthcare.pipeline.explainable_pipeline"
    )
    pipeline_cls = getattr(module, "ExplainableEnsemblePipeline")

    config = {
        "preprocessing": {
            "tabular": {
                "numeric_imputation": "median",
                "categorical_imputation": "most_frequent",
                "encoding_method": "onehot",
                "scaling_method": "standard",
            }
        }
    }
    pipeline = pipeline_cls(
        config=config,
        random_state=7,
        background_sample=12,
        top_k=3,
    )
    pipeline.fit_from_dataframe(
        df,
        target_column="target",
        class_weight=None,
        use_smote=False,
    )
    artifact_path = pipeline.save(tmp_dir)

    sample_row = df.drop(columns=["target"]).iloc[0]
    sample_features: Dict[str, float] = {
        col: float(sample_row[col]) for col in sample_row.index
    }

    original_path = os.environ.get("PIPELINE_ARTIFACT_PATH")
    os.environ["PIPELINE_ARTIFACT_PATH"] = str(artifact_path)
    original_version = os.environ.get("APP_VERSION")
    os.environ["APP_VERSION"] = "test-1.0.0"

    yield {
        "features": sample_features,
        "artifact": artifact_path,
    }

    if original_path is None:
        os.environ.pop("PIPELINE_ARTIFACT_PATH", None)
    else:
        os.environ["PIPELINE_ARTIFACT_PATH"] = original_path
    if original_version is None:
        os.environ.pop("APP_VERSION", None)
    else:
        os.environ["APP_VERSION"] = original_version


@pytest.fixture(scope="module")
def client(api_setup: Dict[str, Any]) -> TestClient:
    # Reload to pick up fresh environment variables
    from hybrid_xai_healthcare.api import app as app_module  # type: ignore[import]

    importlib.reload(app_module)
    return TestClient(app_module.app)


@pytest.fixture(scope="module")
def sample_features(api_setup: Dict[str, Any]) -> Dict[str, float]:
    return dict(api_setup["features"])


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_ready(client: TestClient) -> None:
    response = client.get("/ready")
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "ready"
    assert "Pipeline loaded" in body["detail"]


def test_predict_pipeline(
    client: TestClient, sample_features: Dict[str, float]
) -> None:
    response = client.post("/predict", json={"features": sample_features})
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] != "stub-0.0.1"
    assert body["prediction"]
    assert body["probabilities"]
    assert pytest.approx(sum(body["probabilities"].values()), rel=1e-4) == 1.0
    assert body["explanations"]["top_features"]
    assert body["similar_patients"]


def test_explain_endpoint(
    client: TestClient, sample_features: Dict[str, float]
) -> None:
    response = client.post("/explain", json={"features": sample_features})
    assert response.status_code == 200
    body = response.json()
    assert body["explanations"]["top_features"]
    assert body["trace_id"]


def test_predict_validation(client: TestClient) -> None:
    response = client.post("/predict", json={"features": {}})
    assert response.status_code == 400
