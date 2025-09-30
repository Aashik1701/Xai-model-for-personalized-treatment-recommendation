"""API package for hybrid_xai_healthcare.

Provides FastAPI application exposing prediction, explanation, and health endpoints.
This initial scaffold includes:
  - Health & readiness endpoints
  - Version endpoint
  - Stub prediction & explanation endpoints returning fixed structure
Future enhancements will load a model artifact and reuse preprocessing pipeline.
"""
