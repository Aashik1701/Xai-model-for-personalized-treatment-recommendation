"""Pydantic schemas for API requests and responses."""
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall service status")
    detail: Optional[str] = Field(None, description="Additional context")


class VersionResponse(BaseModel):
    name: str
    version: str
    model_registry: Optional[str] = None


class PredictionRequest(BaseModel):
    # Generic feature mapping. Later: enforce from config/data_config.yaml
    features: Dict[str, Any] = Field(
        ..., description="Feature name to value mapping"
    )
    request_id: Optional[str] = Field(
        None, description="Client-provided id for tracing"
    )


class Explanation(BaseModel):
    feature: str
    value: float


class PredictionResponse(BaseModel):
    prediction: str
    probabilities: Dict[str, float]
    explanations: Dict[str, Any]
    similar_patients: List[str]
    trace_id: str
    model_version: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
