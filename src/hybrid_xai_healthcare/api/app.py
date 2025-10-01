"""FastAPI application exposing prediction, explanation, and health endpoints.

Features:
  - Lazy loading of an :class:`ExplainableEnsemblePipeline` artifact saved by the
    training pipeline (joblib format)
  - Automatic preprocessing using the persisted ``HealthcareDataPreprocessor``
  - SHAP-based local explanations when the pipeline artifact includes a SHAP
    explainer (see :mod:`hybrid_xai_healthcare.pipeline.explainable_pipeline`)
  - Graceful degradation to deterministic stub responses when an artifact is not
    available, preserving availability for smoke tests

Environment variables:
  ``PIPELINE_ARTIFACT_PATH``
      Optional absolute/relative path to ``explainable_ensemble.joblib``.
      Defaults to ``models/checkpoints/explainable_ensemble.joblib``.
  ``MODEL_REGISTRY_URI``
      Display-only metadata referencing the MLflow registry location.
  ``APP_VERSION``
      Overrides the reported application version.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from fastapi import FastAPI, HTTPException

from hybrid_xai_healthcare.pipeline.explainable_pipeline import (
    ExplainableEnsemblePipeline,
)

from .schemas import (
    ErrorResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    VersionResponse,
)

logger = logging.getLogger(__name__)

SERVICE_NAME = "hybrid-xai-healthcare-api"
SERVICE_VERSION = os.getenv("APP_VERSION", "0.1.0")
MODEL_REGISTRY_URI = os.getenv("MODEL_REGISTRY_URI", "mlruns")
DEFAULT_ARTIFACT = Path("models/checkpoints/explainable_ensemble.joblib")

app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION)


class PipelineService:
    """Encapsulates lazy loading and refreshing of the ensemble pipeline."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pipeline: Optional[ExplainableEnsemblePipeline] = None
        self._artifact_path: Optional[Path] = None
        self._artifact_mtime: Optional[float] = None
        self.model_version: str = "unloaded"

    # ------------------------------------------------------------------
    def _resolve_artifact_path(self) -> Path:
        raw_path = os.getenv("PIPELINE_ARTIFACT_PATH")
        if raw_path:
            path = Path(raw_path)
        else:
            path = DEFAULT_ARTIFACT
        return path.expanduser().resolve()

    def _load_pipeline(self, path: Path) -> ExplainableEnsemblePipeline:
        logger.info("Loading pipeline artifact from %s", path)
        pipeline = ExplainableEnsemblePipeline.load(path)
        metadata = (
            pipeline.training_metrics
            if isinstance(pipeline.training_metrics, dict)
            else {}
        )
        version = (
            metadata.get("model_version")
            or metadata.get("version")
            or pipeline.best_model_name
            or "unknown"
        )
        self.model_version = str(version)
        self._pipeline = pipeline
        self._artifact_path = path
        self._artifact_mtime = path.stat().st_mtime
        return pipeline

    def _artifact_changed(self, path: Path) -> bool:
        if self._artifact_mtime is None:
            return True
        try:
            current_mtime = path.stat().st_mtime
        except FileNotFoundError:
            logger.debug("Artifact %s missing during change check", path)
            raise
        return not np.isclose(current_mtime, self._artifact_mtime, atol=1e-6)

    def get_pipeline(self) -> ExplainableEnsemblePipeline:
        path = self._resolve_artifact_path()
        if not path.exists():
            raise FileNotFoundError(f"Pipeline artifact not found at {path}")

        with self._lock:
            if self._pipeline is None:
                return self._load_pipeline(path)
            if self._artifact_path != path or self._artifact_changed(path):
                return self._load_pipeline(path)
            return self._pipeline

    def is_available(self) -> bool:
        try:
            self.get_pipeline()
            return True
        except FileNotFoundError:
            logger.warning("Pipeline artifact not found; API will use stub predictions")
            return False
        except Exception as exc:  # noqa: BLE001
            logger.exception("Pipeline loading failed: %s", exc)
            return False

    def expected_features(self) -> List[str]:
        if not self._pipeline:
            return []
        preprocessor = getattr(self._pipeline, "preprocessor", None)
        names = getattr(preprocessor, "feature_names_in_", None)
        if names:
            return list(names)
        return []

    def preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        pipeline = self.get_pipeline()
        preprocessor = getattr(pipeline, "preprocessor", None)
        if preprocessor is None:
            raise RuntimeError("Loaded pipeline missing preprocessor")
        transformed = preprocessor.transform(df)
        return df, np.asarray(transformed)


pipeline_service = PipelineService()


def _prepare_dataframe(
    features: Dict[str, Any],
    expected: List[str],
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    frame = pd.DataFrame([features])
    if not expected:
        return frame, [], []
    missing = [col for col in expected if col not in frame.columns]
    for col in missing:
        frame[col] = None
    extra = [col for col in frame.columns if col not in expected]
    frame = frame[[col for col in expected]]
    return frame, missing, extra


def _compute_similarity(
    pipeline: ExplainableEnsemblePipeline,
    X_proc: np.ndarray,
) -> List[str]:
    background = getattr(pipeline, "background_data", None)
    if background is None or len(background) == 0 or X_proc.size == 0:
        return []
    try:
        distances = np.linalg.norm(background - X_proc[0], axis=1)
        order = np.argsort(distances)[:2]
        return [f"SIM-{int(idx)}" for idx in order]
    except Exception:  # noqa: BLE001
        return []


def _prediction_from_pipeline(features: Dict[str, Any]) -> PredictionResponse:
    pipeline = pipeline_service.get_pipeline()
    expected = pipeline_service.expected_features()
    df, missing, extra = _prepare_dataframe(features, expected)

    _, transformed = pipeline_service.preprocess(df)
    result = pipeline.predict(df)

    predictions = result.get("predictions", [])
    if not predictions:
        raise RuntimeError("Pipeline returned no predictions")
    predicted = predictions[0]

    classes = result.get("classes") or pipeline.class_labels
    prob_map: Dict[str, float] = {}
    proba = result.get("probabilities")
    if proba is not None:
        first_row = proba[0] if isinstance(proba, list) else proba
        for idx, cls in enumerate(classes):
            try:
                prob_map[str(cls)] = float(first_row[idx])
            except Exception:  # noqa: BLE001
                continue

    explanations_list = result.get("explanations") or []
    explanation = explanations_list[0] if explanations_list else {}
    explanation_payload: Dict[str, Any] = {
        "top_features": explanation.get("top_features", []),
        "predicted_class": explanation.get("predicted_class", predicted),
    }
    if "base_value" in explanation:
        explanation_payload["base_value"] = explanation["base_value"]
    if "probabilities" in explanation:
        explanation_payload["class_probabilities"] = explanation["probabilities"]
    notes: List[str] = []
    if missing:
        notes.append(f"Filled missing features with None: {missing}")
    if extra:
        notes.append(f"Ignored extra features: {extra}")
    if notes:
        explanation_payload["notes"] = notes

    similar = _compute_similarity(pipeline, transformed)
    if not similar:
        hash_sig = sum(hash(str(v)) for v in features.values())
        similar = [f"PID-{(hash_sig % 9000) + 1000}"]

    trace = str(uuid.uuid4())
    return PredictionResponse(
        prediction=str(predicted),
        probabilities=prob_map,
        explanations=explanation_payload,
        similar_patients=similar,
        trace_id=trace,
        model_version=pipeline_service.model_version,
    )


def _mock_predict(features: Dict[str, Any]) -> PredictionResponse:
    classes = ["treatment_A", "treatment_B", "treatment_C"]
    h = sum(hash(str(v)) for v in features.values())
    base = [abs(hash(c) + h) % 100 + 1 for c in classes]
    total = float(sum(base))
    probs = {c: round(b / total, 4) for c, b in zip(classes, base)}
    top = max(list(probs.keys()), key=lambda k: probs[k])
    explanation = {
        "top_features": [
            {"feature": name, "contribution": round((idx + 1) * 0.01, 4)}
            for idx, name in enumerate(list(features.keys())[:5])
        ],
        "predicted_class": top,
        "notes": ["Stub response - model artifact unavailable"],
    }
    similar = [f"PID-{(h % 5000) + i}" for i in range(1, 3)]
    return PredictionResponse(
        prediction=top,
        probabilities=probs,
        explanations=explanation,
        similar_patients=similar,
        trace_id=str(uuid.uuid4()),
        model_version="stub-0.0.1",
    )


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    status = "ok"
    detail = "Service healthy"
    return HealthResponse(status=status, detail=detail)


@app.get("/ready", response_model=HealthResponse, tags=["meta"])
async def readiness() -> HealthResponse:
    if pipeline_service.is_available():
        return HealthResponse(
            status="ready",
            detail=f"Pipeline loaded ({pipeline_service.model_version})",
        )
    return HealthResponse(
        status="degraded",
        detail="Pipeline artifact unavailable; using stub",
    )


@app.get("/version", response_model=VersionResponse, tags=["meta"])
async def version() -> VersionResponse:
    return VersionResponse(
        name=SERVICE_NAME,
        version=SERVICE_VERSION,
        model_registry=MODEL_REGISTRY_URI,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["inference"],
)
async def predict(req: PredictionRequest) -> PredictionResponse:
    if not req.features:
        raise HTTPException(status_code=400, detail="No features provided")
    if pipeline_service.is_available():
        try:
            return _prediction_from_pipeline(req.features)
        except FileNotFoundError:
            logger.warning(
                "Pipeline disappeared between availability check and prediction"
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Pipeline prediction failed: %s", exc)
            raise HTTPException(status_code=500, detail="Pipeline prediction failed")
    return _mock_predict(req.features)


@app.post(
    "/explain",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["inference"],
)
async def explain(req: PredictionRequest) -> PredictionResponse:
    if not req.features:
        raise HTTPException(status_code=400, detail="No features provided")
    if pipeline_service.is_available():
        try:
            return _prediction_from_pipeline(req.features)
        except FileNotFoundError:
            logger.warning(
                "Pipeline disappeared between availability check and explanation"
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Pipeline explanation failed: %s", exc)
            raise HTTPException(status_code=500, detail="Pipeline explanation failed")
    return _mock_predict(req.features)


@app.get("/", include_in_schema=False)
async def root() -> Dict[str, str]:  # pragma: no cover - trivial
    return {"message": "Hybrid XAI Healthcare API"}
