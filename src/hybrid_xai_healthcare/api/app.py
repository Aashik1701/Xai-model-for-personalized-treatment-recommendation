"""FastAPI application exposing prediction, explanation, and health endpoints.

This is an initial scaffold. It currently:
  - Provides health, readiness, and version endpoints
  - Implements a stub prediction endpoint returning deterministic mock values
  - Generates a trace_id for each request
  - Outlines where model loading & preprocessing integration will occur

Future work:
  - Load latest Production model from MLflow model registry
  - Apply preprocessing pipeline identical to training
  - Compute real explanations (SHAP/LIME) and similarity results
  - Add authentication / authorization middleware
  - Observability: metrics, structured logging, tracing (OpenTelemetry)
"""

from __future__ import annotations

import os
import uuid
from functools import lru_cache
from typing import Dict, Any
from fastapi import FastAPI, HTTPException

from .schemas import (
    HealthResponse,
    VersionResponse,
    PredictionRequest,
    PredictionResponse,
    ErrorResponse,
)

SERVICE_NAME = "hybrid-xai-healthcare-api"
SERVICE_VERSION = os.getenv("APP_VERSION", "0.1.0")
MODEL_REGISTRY_URI = os.getenv("MODEL_REGISTRY_URI", "mlruns")

app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION)


@lru_cache(maxsize=1)
def get_model_stub() -> Dict[str, Any]:
    # Placeholder for a future model object (e.g., MLflow pyfunc or sklearn pipeline)
    return {
        "model_version": "stub-0.0.1",
        "classes": ["treatment_A", "treatment_B", "treatment_C"],
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok", detail="Service healthy")


@app.get("/ready", response_model=HealthResponse, tags=["meta"])
async def readiness() -> HealthResponse:
    # In future, check model loaded, dependencies, external services
    _ = get_model_stub()
    return HealthResponse(status="ready", detail="Model stub loaded")


@app.get("/version", response_model=VersionResponse, tags=["meta"])
async def version() -> VersionResponse:
    return VersionResponse(
        name=SERVICE_NAME,
        version=SERVICE_VERSION,
        model_registry=MODEL_REGISTRY_URI,
    )


def _mock_predict(features: Dict[str, Any]) -> PredictionResponse:
    model_info = get_model_stub()
    classes = model_info["classes"]
    # Simple deterministic pseudo-probabilities based on hash
    h = sum(hash(str(v)) for v in features.values())
    base = [abs(hash(c) + h) % 100 + 1 for c in classes]
    total = float(sum(base))
    probs = {c: round(b / total, 4) for c, b in zip(classes, base)}
    # Explicit list to satisfy type checkers
    top = max(list(probs.keys()), key=lambda k: probs[k])
    explanations = {
        "feature_importances": [
            {"feature": k, "shap_value": round((i + 1) * 0.01, 4)}
            for i, k in enumerate(list(features.keys())[:5])
        ],
        "smote_applied": False,
    }
    similar = [f"PID-{(h % 5000) + i}" for i in range(1, 3)]
    trace = str(uuid.uuid4())
    return PredictionResponse(
        prediction=top,
        probabilities=probs,
        explanations=explanations,
        similar_patients=similar,
        trace_id=trace,
        model_version=model_info["model_version"],
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}},
    tags=["inference"],
)
async def predict(req: PredictionRequest) -> PredictionResponse:
    if not req.features:
        raise HTTPException(status_code=400, detail="No features provided")
    return _mock_predict(req.features)


# Alias for future explanation-specific endpoint (for now bundled in prediction)
@app.post(
    "/explain",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}},
    tags=["inference"],
)
async def explain(req: PredictionRequest) -> PredictionResponse:
    if not req.features:
        raise HTTPException(status_code=400, detail="No features provided")
    return _mock_predict(req.features)


@app.get("/", include_in_schema=False)
async def root() -> Dict[str, str]:  # pragma: no cover - trivial
    return {"message": "Hybrid XAI Healthcare API"}
