"""Tests for the FastAPI inference scaffold."""

from fastapi.testclient import TestClient
from hybrid_xai_healthcare.api.app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_version():
    r = client.get("/version")
    assert r.status_code == 200
    assert "version" in r.json()


def test_predict_stub():
    payload = {"features": {"age": 65, "cholesterol": 210, "bp": 130}}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "prediction" in body
    assert "probabilities" in body
    assert "trace_id" in body
    assert len(body["probabilities"]) >= 1


def test_predict_validation():
    r = client.post("/predict", json={"features": {}})
    assert r.status_code == 400
