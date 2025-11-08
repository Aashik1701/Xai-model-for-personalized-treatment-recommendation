"""
Production FastAPI Application for ML Model Serving

This API provides endpoints for:
- Single and batch predictions
- Model explanations (SHAP)
- Model listing and health checks
- Performance metrics

Usage:
    uvicorn api.production_api:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import time
import uuid
import io
import sys
from pathlib import Path
import os
import tempfile

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add scripts to path for imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

# Import explainability and personalization
try:
    from explainability_toolkit import ExplainabilityToolkit
    import shap

    EXPLAINABILITY_AVAILABLE = True
    logger.info("✅ Explainability toolkit loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Explainability toolkit not available: {e}")
    EXPLAINABILITY_AVAILABLE = False

try:
    from personalization_engine import PersonalizationEngine

    PERSONALIZATION_AVAILABLE = True
    logger.info("✅ Personalization engine loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Personalization engine not available: {e}")
    PERSONALIZATION_AVAILABLE = False

# MLflow configuration
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Initialize FastAPI app
app = FastAPI(
    title="Healthcare ML Prediction API",
    description="Production API for explainable healthcare AI models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
MODEL_CACHE: Dict[str, Any] = {}


# Request/Response Models
class PatientData(BaseModel):
    """Single patient data for prediction"""

    features: Dict[str, float] = Field(
        ..., description="Patient features as key-value pairs"
    )

    class Config:
        schema_extra = {
            "example": {
                "features": {
                    "age": 65.0,
                    "gender": 1.0,
                    "blood_pressure": 140.0,
                    "cholesterol": 200.0,
                }
            }
        }


class PredictionRequest(BaseModel):
    """Prediction request with optional explainability"""

    model_name: str = Field(..., description="Registered model name")
    patient_data: PatientData
    return_probabilities: bool = Field(True, description="Return class probabilities")
    include_explanation: bool = Field(False, description="Include SHAP explanation")
    include_personalization: bool = Field(False, description="Include similar patients")

    @validator("model_name")
    def validate_model_name(cls, v):
        available_models = [
            "SyntheaLongitudinal",
            "ISICSkinCancer",
            "DrugReviews",
            "BreastCancer",
            "HeartDisease",
        ]
        if v not in available_models:
            raise ValueError(f"Model must be one of: {available_models}")
        return v


class PredictionResponse(BaseModel):
    """Prediction response with optional explanations"""

    prediction: Any
    probabilities: Optional[Dict[str, float]] = None
    confidence: float
    model_name: str
    model_version: str
    inference_time_ms: float
    trace_id: str
    timestamp: str
    explanation: Optional[Dict[str, Any]] = None  # SHAP values & feature importance
    personalization: Optional[Dict[str, Any]] = None  # Similar patients & cohort info


class ExplainRequest(BaseModel):
    """Explanation request"""

    model_name: str
    patient_data: PatientData
    explanation_type: str = Field("shap", description="Type: 'shap' or 'lime'")
    include_personalization: bool = Field(False, description="Include similar patients")


class PersonalizeRequest(BaseModel):
    """Personalization request"""

    patient_data: PatientData
    model_name: str
    k_neighbors: int = Field(5, description="Number of similar patients to find")
    include_cohort: bool = Field(True, description="Include cohort assignment")


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    timestamp: str
    mlflow_connected: bool
    models_loaded: int
    uptime_seconds: float


# Startup event
START_TIME = time.time()

# Global instances for explainability and personalization
explainability_toolkit = None
personalization_engine = None
model_cache = {}  # Cache for loaded models and explainers
# Per-model background cache to improve SHAP stability
BACKGROUND_CACHE: Dict[str, List[np.ndarray]] = {}
# Optional persisted background per model (loaded from MLflow artifacts or disk)
BACKGROUND_ARTIFACT_CACHE: Dict[str, np.ndarray] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_background_from_input(
    input_df: pd.DataFrame, n: int = 128, scale: float = 0.2
) -> np.ndarray:
    """Create a synthetic background distribution around the input.

    This helps SHAP approximate expectations when we don't have training data on hand.
    Args:
        input_df: single-row DataFrame of the input features
        n: number of background samples
        scale: relative noise level (as a fraction of |x|); min absolute noise 0.1
    Returns:
        ndarray of shape (n, n_features)
    """
    x0 = input_df.values[0]
    rng = np.random.default_rng(42)
    noise_scale = np.maximum(np.abs(x0) * scale, 0.1)
    background = np.vstack([x0 + rng.normal(0, noise_scale) for _ in range(n)])
    return background


def _align_features(input_df: pd.DataFrame, sklearn_model: Any) -> pd.DataFrame:
    """Align input columns to model's training feature order if available.

    - Reorders columns to match feature_names_in_
    - Drops unknown columns
    - Fills any missing expected columns with 0.0
    """
    try:
        if hasattr(sklearn_model, "feature_names_in_"):
            expected = list(sklearn_model.feature_names_in_)
            # Keep only expected features
            df = input_df.reindex(columns=expected, fill_value=0.0)
            # Ensure numeric dtype
            return df.astype(float)
    except Exception:
        pass
    return input_df


def _validate_input_features(input_df: pd.DataFrame, sklearn_model: Any) -> None:
    """Validate that the request includes the full expected feature set.

    Raises HTTPException(400) with details on mismatch.
    """
    expected: Optional[List[str]] = None
    # If model is a pipeline, try to get expected input from preprocessor
    try:
        if hasattr(sklearn_model, "named_steps"):
            steps = getattr(sklearn_model, "named_steps", {})
            pre = steps.get("preprocessor")
            if pre is not None and hasattr(pre, "feature_names_in_"):
                expected = list(pre.feature_names_in_)
    except Exception:
        expected = None

    # Fallback to model.feature_names_in_
    if expected is None and hasattr(sklearn_model, "feature_names_in_"):
        try:
            expected = list(sklearn_model.feature_names_in_)
        except Exception:
            expected = None

    if expected is None:
        # Fall back to n_features_in_ if available
        try:
            if hasattr(sklearn_model, "n_features_in_"):
                expected_count = int(getattr(sklearn_model, "n_features_in_"))
                provided_count = len(input_df.columns)
                if provided_count != expected_count:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "Feature count mismatch. "
                            f"expected={expected_count}, got={provided_count}"
                        ),
                    )
                return
        except Exception:
            pass
        # No schema metadata available; skip strict check
        return

    provided = list(input_df.columns)
    missing = [c for c in expected if c not in provided]
    extra = [c for c in provided if c not in expected]
    if missing or extra or (len(provided) != len(expected)):
        raise HTTPException(
            status_code=400,
            detail=(
                "Feature schema mismatch. "
                f"expected={len(expected)}, got={len(provided)}; "
                f"missing={missing}; extra={extra}"
            ),
        )


def _get_pipeline_parts(sklearn_model: Any):
    """Return (preprocessor, classifier, is_pipeline) for sklearn pipelines."""
    try:
        if hasattr(sklearn_model, "named_steps") and isinstance(
            getattr(sklearn_model, "named_steps"), dict
        ):
            steps = sklearn_model.named_steps
            pre = steps.get("preprocessor")
            clf = steps.get("classifier")
            if clf is None and len(steps) > 0:
                # Assume last step is the estimator
                last_key = list(steps.keys())[-1]
                clf = steps[last_key]
            return pre, clf, True
    except Exception:
        pass
    return None, sklearn_model, False


def _load_background_artifact_for_model(
    model_name: str, expected_cols: Optional[List[str]] = None, max_rows: int = 256
) -> Optional[np.ndarray]:
    """Try to load a persisted background from MLflow artifacts for this model.

    Looks for common file names like background.npy or background.csv (optionally
    under a background/ subfolder) in the run artifacts of the Production version.
    """
    try:
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        prod = next((v for v in versions if v.current_stage == "Production"), None)
        if not prod or not getattr(prod, "run_id", None):
            return None

        run_id = prod.run_id
        candidates = [
            "background.npy",
            "background/background.npy",
            "background.csv",
            "background/background.csv",
            "background.parquet",
            "background/background.parquet",
        ]

        tmpdir = tempfile.mkdtemp(prefix="bgdl_")
        local_path = None

        # Try client.download_artifacts first (compatible across mlflow versions)
        for path in candidates:
            try:
                res = client.download_artifacts(run_id, path, tmpdir)
                if res and os.path.exists(res):
                    local_path = res
                    break
            except Exception:
                continue

        if local_path is None:
            return None

        arr: Optional[np.ndarray] = None
        if local_path.endswith(".npy"):
            arr = np.load(local_path)
        elif local_path.endswith(".csv"):
            df = pd.read_csv(local_path)
            if expected_cols:
                df = df.reindex(columns=expected_cols, fill_value=0.0)
            arr = df.values
        elif local_path.endswith(".parquet"):
            df = pd.read_parquet(local_path)
            if expected_cols:
                df = df.reindex(columns=expected_cols, fill_value=0.0)
            arr = df.values

        if arr is None or arr.ndim != 2 or arr.shape[0] == 0:
            return None

        # Subsample for performance
        if arr.shape[0] > max_rows:
            rng = np.random.default_rng(123)
            idx = rng.choice(arr.shape[0], size=max_rows, replace=False)
            arr = arr[idx]

        BACKGROUND_ARTIFACT_CACHE[model_name] = arr
        logger.info(f"Loaded background artifact for {model_name}: shape={arr.shape}")
        return arr
    except Exception:
        return None


def _update_background_cache(
    model_name: str, input_df: pd.DataFrame, max_keep: int = 512
) -> None:
    """Append input to a per-model background cache."""
    try:
        x0 = np.array(input_df.values[0], dtype=float)
        arr = BACKGROUND_CACHE.get(model_name, [])
        arr.append(x0)
        if len(arr) > max_keep:
            arr[:] = arr[-max_keep:]
        BACKGROUND_CACHE[model_name] = arr
    except Exception:
        pass


def _get_background_for_model(model_name: str, input_df: pd.DataFrame) -> np.ndarray:
    """Return representative background for SHAP."""
    # Prefer a persisted artifact background if available or loadable
    if (
        model_name in BACKGROUND_ARTIFACT_CACHE
        and BACKGROUND_ARTIFACT_CACHE[model_name] is not None
    ):
        arr_art = BACKGROUND_ARTIFACT_CACHE[model_name]
        return arr_art

    # Try to load from MLflow artifacts on first use
    loaded = _load_background_artifact_for_model(
        model_name, expected_cols=list(input_df.columns)
    )
    if loaded is not None:
        return loaded

    # Else, use cached recent inputs with small jitter
    arr = BACKGROUND_CACHE.get(model_name, [])
    if len(arr) >= 64:
        rng = np.random.default_rng(123)
        sample_sz = min(192, len(arr))
        idx = rng.choice(len(arr), size=sample_sz, replace=False)
        chosen = np.stack([arr[i] for i in idx], axis=0)
        noise = np.maximum(np.abs(chosen) * 0.1, 0.05)
        jitter = rng.normal(0, noise)
        return chosen + jitter
    return _build_background_from_input(input_df, n=160, scale=0.25)


def _extract_sklearn_model_from_pyfunc(pyfunc_model: Any):
    """Return an sklearn estimator from a loaded pyfunc model if available.

    This handles cases where the pyfunc was created via a PythonModel wrapper
    that holds the true sklearn estimator under `python_model.model`.
    """
    try:
        impl = getattr(pyfunc_model, "_model_impl", None)
        if impl is None:
            return None
        skl = getattr(impl, "sklearn_model", None)
        if skl is not None:
            return skl
        py = getattr(impl, "python_model", None)
        if py is not None and hasattr(py, "model"):
            return py.model
    except Exception:
        return None
    return None


# For cohort semantics, specify whether class 1 means "higher risk" (True)
# or "favorable" (False). Adjust here for your registered models.
RISK_POSITIVE_CLASS_MEANS_HIGH_RISK: Dict[str, bool] = {
    "HeartDisease": True,
    "BreastCancer": True,
    "ISICSkinCancer": True,
    "SyntheaLongitudinal": True,
    # For DrugReviews, class 1 often means high rating (favorable), so invert
    "DrugReviews": False,
}


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    global explainability_toolkit, personalization_engine

    logger.info("=" * 80)
    logger.info("🚀 Starting Healthcare ML Prediction API...")
    logger.info("=" * 80)
    logger.info(f"MLflow URI: {MLFLOW_TRACKING_URI}")

    # Verify MLflow connection
    try:
        client = MlflowClient()
        models = client.search_registered_models()
        logger.info(f"✅ Found {len(models)} registered models in MLflow")
    except Exception as e:
        logger.error(f"❌ Failed to connect to MLflow: {e}")

    # Initialize explainability toolkit
    if EXPLAINABILITY_AVAILABLE:
        try:
            explainability_toolkit = ExplainabilityToolkit()
            logger.info("✅ Explainability toolkit initialized")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize explainability: {e}")
            explainability_toolkit = None
    else:
        logger.warning("⚠️ Explainability features disabled (SHAP/LIME not available)")

    # Initialize personalization engine
    if PERSONALIZATION_AVAILABLE:
        try:
            personalization_engine = PersonalizationEngine()
            logger.info("✅ Personalization engine initialized")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize personalization: {e}")
            personalization_engine = None
    else:
        logger.warning("⚠️ Personalization features disabled")

    logger.info("=" * 80)
    logger.info("🎯 API Ready!")
    logger.info("   - Predictions: ✅ Enabled")
    explain_status = "✅ Enabled" if explainability_toolkit else "❌ Disabled"
    logger.info(f"   - Explainability: {explain_status}")
    pers_status = "✅ Enabled" if personalization_engine else "❌ Disabled"
    logger.info(f"   - Personalization: {pers_status}")
    logger.info("=" * 80)


def load_model(model_name: str, stage: str = "Production"):
    """Load model from MLflow registry with caching"""
    cache_key = f"{model_name}_{stage}"

    if cache_key in MODEL_CACHE:
        logger.info(f"Loading {model_name} from cache")
        return MODEL_CACHE[cache_key]

    try:
        model_uri = f"models:/{model_name}/{stage}"
        logger.info(f"Loading model: {model_uri}")

        start_time = time.time()
        model = mlflow.pyfunc.load_model(model_uri)
        load_time = time.time() - start_time

        logger.info(f"Model loaded in {load_time:.2f}s")

        # If the pyfunc model wraps a python_model that itself contains a sklearn estimator
        # (for example our pyfunc wrapper around joblib dicts), normalize by exposing
        # sklearn_model on the internal impl so downstream code (which expects
        # model._model_impl.sklearn_model) continues to work.
        try:
            impl = getattr(model, "_model_impl", None)
            if impl is not None and getattr(impl, "sklearn_model", None) is None:
                py = getattr(impl, "python_model", None)
                if py is not None and hasattr(py, "model"):
                    impl.sklearn_model = py.model
                    logger.info(
                        "Normalized pyfunc model: exposed sklearn_model from python_model.model"
                    )
        except Exception:
            pass
        # Cache the model
        MODEL_CACHE[cache_key] = {
            "model": model,
            "loaded_at": datetime.now().isoformat(),
            "load_time_seconds": load_time,
        }

        return MODEL_CACHE[cache_key]

    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Healthcare ML Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    mlflow_connected = False

    try:
        client = MlflowClient()
        client.search_registered_models()
        mlflow_connected = True
    except Exception as e:
        logger.error(f"MLflow health check failed: {e}")

    return HealthResponse(
        status="healthy" if mlflow_connected else "degraded",
        timestamp=datetime.now().isoformat(),
        mlflow_connected=mlflow_connected,
        models_loaded=len(MODEL_CACHE),
        uptime_seconds=time.time() - START_TIME,
    )


@app.get("/models")
async def list_models():
    """List all available models"""
    try:
        client = MlflowClient()
        models = client.search_registered_models()

        model_list = []
        for model in models:
            production_versions = [
                v for v in model.latest_versions if v.current_stage == "Production"
            ]

            if production_versions:
                version = production_versions[0]

                # Get tags
                tags = {t.key: t.value for t in version.tags}

                model_list.append(
                    {
                        "name": model.name,
                        "version": version.version,
                        "stage": version.current_stage,
                        "description": model.description,
                        "accuracy": tags.get("accuracy", "N/A"),
                        "use_case": tags.get("use_case", "N/A"),
                        "run_id": version.run_id,
                    }
                )

        return {
            "models": model_list,
            "total": len(model_list),
            "registry_uri": MLFLOW_TRACKING_URI,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@app.get("/_debug/model_info")
async def debug_model_info(model_name: str):
    """Debug endpoint: return introspection info for a registered model (non-production use).

    Returns the top-level type, whether pyfunc is present, and attributes of the underlying
    python_model/sklearn estimator when available.
    """
    try:
        model_data = load_model(model_name)
        model = model_data["model"]
        info = {"model_name": model_name, "top_type": str(type(model))}

        # Inspect pyfunc internals if available
        impl = getattr(model, "_model_impl", None)
        info["has__model_impl"] = impl is not None
        if impl is not None:
            py = getattr(impl, "python_model", None)
            skl = getattr(impl, "sklearn_model", None)
            info["has_python_model"] = py is not None
            info["has_sklearn_model"] = skl is not None
            info["python_model_type"] = str(type(py)) if py is not None else None
            info["sklearn_model_type"] = str(type(skl)) if skl is not None else None

            # if python_model has 'model' attribute, show its type
            if py is not None and hasattr(py, "model"):
                try:
                    info["python_model.inner_model_type"] = str(type(py.model))
                    info["python_model.inner_has_predict"] = hasattr(
                        py.model, "predict"
                    )
                except Exception:
                    pass

        return info
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug inspect failed: {e}")


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction for a single patient"""
    trace_id = str(uuid.uuid4())
    # start_time reserved (not used): using per-step timers below

    try:
        # Load model
        model_data = load_model(request.model_name)
        model = model_data["model"]

        # Prepare input data
        input_df = pd.DataFrame([request.patient_data.features])
        # Validate and align feature schema before prediction
        try:
            if hasattr(model, "_model_impl") and hasattr(
                model._model_impl, "sklearn_model"
            ):
                sk = model._model_impl.sklearn_model
                _validate_input_features(input_df, sk)
                input_df = _align_features(input_df, sk)
        except HTTPException:
            raise
        except Exception:
            # If schema metadata is unavailable, proceed (model may still accept)
            pass
        # Align feature order to training if available
        try:
            if hasattr(model, "_model_impl") and hasattr(
                model._model_impl, "sklearn_model"
            ):
                input_df = _align_features(input_df, model._model_impl.sklearn_model)
        except Exception:
            pass

        # Make prediction
        prediction_start = time.time()
        prediction = model.predict(input_df)

        # Try to get probabilities (if classifier)
        probabilities = None
        confidence = 1.0

        try:
            # For sklearn models wrapped in pyfunc
            if hasattr(model, "_model_impl") and hasattr(
                model._model_impl, "sklearn_model"
            ):
                sklearn_model = model._model_impl.sklearn_model
                if hasattr(sklearn_model, "predict_proba"):
                    proba = sklearn_model.predict_proba(input_df)[0]
                    probabilities = {
                        f"class_{i}": float(p) for i, p in enumerate(proba)
                    }
                    confidence = float(max(proba))
        except Exception as e:
            logger.warning(f"Could not get probabilities: {e}")

        inference_time = (time.time() - prediction_start) * 1000  # ms

        # Get model version
        client = MlflowClient()
        model_versions = client.search_model_versions(f"name='{request.model_name}'")
        production_version = next(
            (v.version for v in model_versions if v.current_stage == "Production"),
            "unknown",
        )

        # Generate explanation if requested
        explanation_result = None
        if (
            request.include_explanation
            and EXPLAINABILITY_AVAILABLE
            and explainability_toolkit
        ):
            try:
                logger.info(f"[{trace_id}] Generating SHAP explanation...")

                sklearn_model = model._model_impl.sklearn_model
                # Validate full feature schema (strict) and align ordering
                _validate_input_features(input_df, sklearn_model)
                input_df = _align_features(input_df, sklearn_model)
                cache_key = f"shap_explainer_{request.model_name}"

                # Determine if model is a sklearn Pipeline
                pre, clf, is_pipe = _get_pipeline_parts(sklearn_model)

                if is_pipe and pre is not None:
                    # Build background in original feature space then transform
                    bg_orig = _get_background_for_model(request.model_name, input_df)
                    bg_df = pd.DataFrame(bg_orig, columns=list(input_df.columns))
                    bg_trans = pre.transform(bg_df)
                    masker = shap.maskers.Independent(bg_trans)
                    model_fn = clf.predict_proba
                else:
                    # Build a robust background and masker in original space
                    background = _build_background_from_input(
                        input_df, n=160, scale=0.25
                    )
                    masker = shap.maskers.Independent(background)
                    model_fn = sklearn_model.predict_proba

                # Create/cache a general-purpose explainer
                if cache_key not in model_cache:
                    explainer = shap.Explainer(
                        model_fn, masker, algorithm="auto", link="logit"
                    )
                    model_cache[cache_key] = explainer
                    explainer_type = "Explainer(auto/logit)"
                else:
                    explainer = model_cache[cache_key]
                    explainer_type = "Explainer(cache)"

                if is_pipe and pre is not None:
                    X_trans = pre.transform(input_df)
                    sv_obj = explainer(X_trans)
                else:
                    sv_obj = explainer(input_df.values)
                # sv_obj.values shape: (n_samples, n_features, n_outputs)
                values = sv_obj.values

                # Determine class index to explain (predicted class by default)
                class_index = None
                try:
                    proba_tmp = (
                        sklearn_model.predict_proba(input_df)[0]
                        if not is_pipe
                        else sklearn_model.predict_proba(input_df)[0]
                    )
                    class_index = int(np.argmax(proba_tmp))
                except Exception:
                    class_index = 0

                if values.ndim == 3:
                    shap_values = values[0, :, class_index]
                elif values.ndim == 2:
                    shap_values = values[0]
                else:
                    shap_values = values

                # If all zeros, try broadening background once
                if np.allclose(shap_values, 0, atol=1e-9):
                    if is_pipe and pre is not None:
                        bg_orig = _build_background_from_input(
                            input_df, n=256, scale=0.5
                        )
                        bg_df = pd.DataFrame(bg_orig, columns=list(input_df.columns))
                        bg_trans = pre.transform(bg_df)
                        masker = shap.maskers.Independent(bg_trans)
                        explainer_retry = shap.Explainer(
                            clf.predict_proba, masker, algorithm="auto", link="logit"
                        )
                        X_trans = pre.transform(input_df)
                        sv_retry = explainer_retry(X_trans)
                    else:
                        background = _build_background_from_input(
                            input_df, n=256, scale=0.5
                        )
                        masker = shap.maskers.Independent(background)
                        explainer_retry = shap.Explainer(
                            sklearn_model.predict_proba,
                            masker,
                            algorithm="auto",
                            link="logit",
                        )
                        sv_retry = explainer_retry(input_df.values)
                    vals = sv_retry.values
                    if vals.ndim == 3:
                        shap_values = vals[0, :, class_index]
                    elif vals.ndim == 2:
                        shap_values = vals[0]
                    else:
                        shap_values = vals

                # Calculate feature importance (top 5)
                feature_importance = {}
                for i, col in enumerate(input_df.columns):
                    shap_val = float(shap_values[i])
                    feat_val = float(input_df[col].values[0])
                    feature_importance[col] = {
                        "shap_value": shap_val,
                        "feature_value": feat_val,
                        "impact": "positive" if shap_val > 0 else "negative",
                        "magnitude": abs(shap_val),
                    }

                sorted_features = sorted(
                    feature_importance.items(),
                    key=lambda x: x[1]["magnitude"],
                    reverse=True,
                )

                explanation_result = {
                    "method": f"SHAP ({explainer_type})",
                    "top_features": dict(sorted_features[:5]),
                    "total_features": len(feature_importance),
                }

                logger.info("SHAP explanation generated")

            except Exception as e:
                logger.warning(f"⚠️ SHAP explanation failed: {e}")
                # As a last resort, attempt LIME to still provide explanations
                try:
                    from lime.lime_tabular import LimeTabularExplainer

                    sklearn_model = model._model_impl.sklearn_model
                    x0 = input_df.values[0]
                    rng = np.random.default_rng(7)
                    scale = np.maximum(np.abs(x0) * 0.05, 0.01)
                    lime_background = np.vstack(
                        [x0 + rng.normal(0, scale) for _ in range(100)]
                    )

                    lime_explainer = LimeTabularExplainer(
                        training_data=lime_background,
                        feature_names=list(input_df.columns),
                        mode="classification",
                    )
                    exp = lime_explainer.explain_instance(
                        data_row=x0,
                        predict_fn=sklearn_model.predict_proba,
                        num_features=min(10, len(input_df.columns)),
                    )
                    # Map to top_features structure
                    feat_map = {}
                    feat_order = []
                    for feat, weight in exp.as_list():
                        name = feat.split()[0]
                        if name in input_df.columns and name not in feat_map:
                            val = float(input_df[name].values[0])
                            feat_map[name] = {
                                "shap_value": float(weight),  # use LIME weight
                                "feature_value": val,
                                "impact": "positive" if weight > 0 else "negative",
                                "magnitude": abs(float(weight)),
                            }
                            feat_order.append(name)

                    # Sort by magnitude similar to SHAP
                    sorted_features = sorted(
                        feat_map.items(), key=lambda x: x[1]["magnitude"], reverse=True
                    )
                    explanation_result = {
                        "method": "LIME (fallback)",
                        "top_features": dict(sorted_features[:5]),
                        "total_features": len(feat_map),
                    }
                except Exception as le:
                    logger.warning(f"LIME fallback also failed: {le}")
                    explanation_result = {"error": "Explanation generation failed"}

        # Generate personalization if requested
        personalization_result = None
        if (
            request.include_personalization
            and PERSONALIZATION_AVAILABLE
            and personalization_engine
        ):
            try:
                logger.info(f"[{trace_id}] Generating personalization...")

                # Compute risk score aligned with model semantics
                risk_score = float(confidence)
                try:
                    sklearn_model = model._model_impl.sklearn_model
                    if hasattr(sklearn_model, "predict_proba"):
                        proba = sklearn_model.predict_proba(input_df)[0]
                        classes = list(
                            getattr(sklearn_model, "classes_", list(range(len(proba))))
                        )
                        # Default: use class '1' if present, else argmax
                        if 1 in classes:
                            pos_idx = classes.index(1)
                        else:
                            pos_idx = int(np.argmax(proba))
                        p_pos = float(proba[pos_idx])

                        # If positive class means favorable (e.g., high rating),
                        # invert the risk score mapping
                        high_risk_if_positive = RISK_POSITIVE_CLASS_MEANS_HIGH_RISK.get(
                            request.model_name, True
                        )
                        risk_score = p_pos if high_risk_if_positive else (1.0 - p_pos)
                except Exception:
                    pass

                # Determine cohort
                if risk_score >= 0.75:
                    cohort = "high_risk"
                    cohort_description = "High risk - Requires immediate attention"
                elif risk_score >= 0.5:
                    cohort = "medium_risk"
                    cohort_description = "Medium risk - Monitor closely"
                else:
                    cohort = "low_risk"
                    cohort_description = "Low risk - Routine follow-up"

                # Generate treatment recommendations based on top features
                recommendations = []
                if explanation_result and "top_features" in explanation_result:
                    top_features = explanation_result["top_features"]
                    for feature_name, feature_data in list(top_features.items())[:3]:
                        if feature_data["impact"] == "positive":
                            recommendations.append(
                                f"Monitor {feature_name} closely "
                                f"(current: {feature_data['feature_value']})"
                            )

                personalization_result = {
                    "cohort": cohort,
                    "cohort_description": cohort_description,
                    "risk_score": risk_score,
                    "recommendations": recommendations[:3],  # Top 3
                    "similar_patients_found": "feature_coming_soon",
                }

                logger.info(f"Personalization generated - Cohort: {cohort}")

            except Exception as e:
                logger.warning(f"⚠️ Personalization failed: {e}")
                personalization_result = {"error": "Personalization generation failed"}

        # Convert numpy types to Python native types for serialization
        prediction_value = (
            prediction[0] if isinstance(prediction, np.ndarray) else prediction
        )
        if hasattr(prediction_value, "item"):
            prediction_value = prediction_value.item()  # Convert numpy scalar to Python

        response = PredictionResponse(
            prediction=prediction_value,
            probabilities=probabilities if request.return_probabilities else None,
            confidence=float(confidence),
            model_name=request.model_name,
            model_version=str(production_version),
            inference_time_ms=float(inference_time),
            trace_id=trace_id,
            timestamp=datetime.now().isoformat(),
            explanation=explanation_result,
            personalization=personalization_result,
        )

        logger.info(
            f"Prediction completed - Model: {request.model_name}, "
            f"Time: {inference_time:.2f}ms, Trace: {trace_id}"
        )

        return response

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(
    model_name: str,
    file: UploadFile = File(..., description="CSV file with patient data"),
):
    """Make predictions for multiple patients from CSV file"""
    trace_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        # Validate model name
        available_models = [
            "SyntheaLongitudinal",
            "ISICSkinCancer",
            "DrugReviews",
            "BreastCancer",
            "HeartDisease",
        ]
        if model_name not in available_models:
            raise HTTPException(
                status_code=400, detail=f"Model must be one of: {available_models}"
            )

        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        logger.info(f"Batch prediction: {len(df)} samples for {model_name}")

        # Load model
        model_data = load_model(model_name)
        model = model_data["model"]

        # Make predictions
        predictions = model.predict(df)

        # Try to get probabilities
        try:
            if hasattr(model, "_model_impl") and hasattr(
                model._model_impl.python_model, "predict_proba"
            ):
                probabilities = model._model_impl.python_model.predict_proba(df)
                df["confidence"] = probabilities.max(axis=1)
        except Exception:
            df["confidence"] = 1.0

        df["prediction"] = predictions
        df["trace_id"] = trace_id
        df["timestamp"] = datetime.now().isoformat()

        inference_time = (time.time() - start_time) * 1000

        return {
            "predictions": df.to_dict(orient="records"),
            "total_samples": len(df),
            "model_name": model_name,
            "inference_time_ms": inference_time,
            "trace_id": trace_id,
        }

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}"
        )


@app.post("/explain")
async def explain_prediction(request: ExplainRequest):
    """
    Generate SHAP/LIME explanation for a prediction

    Returns feature importance, SHAP values, and optional personalization insights
    """
    trace_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        # Check if explainability is available
        if not EXPLAINABILITY_AVAILABLE or explainability_toolkit is None:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Explainability features not available",
                    "message": (
                        "SHAP/LIME libraries not installed or toolkit "
                        "initialization failed"
                    ),
                    "trace_id": trace_id,
                    "workaround": "Install with: pip install shap lime",
                },
            )

        logger.info(
            f"[{trace_id}] Generating {request.explanation_type} explanation "
            f"for {request.model_name}"
        )

        # Load model from MLflow
        # Use the shared load_model helper which handles caching and
        # normalizing pyfunc-wrapped joblib artifacts (exposing sklearn_model)
        try:
            model_data = load_model(request.model_name)
            model = model_data["model"]
            logger.info(f"✅ Loaded model: {request.model_name}")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Model '{request.model_name}' not found in Production: "
                    f"{str(e)}"
                ),
            )

        # Prepare input data
        input_df = pd.DataFrame([request.patient_data.features])
        # Align feature order to training if available
        try:
            if hasattr(model, "_model_impl") and hasattr(
                model._model_impl, "sklearn_model"
            ):
                input_df = _align_features(input_df, model._model_impl.sklearn_model)
        except Exception:
            pass

        # Generate explanation based on type
        explanation_result = {}

        if request.explanation_type.lower() == "shap":
            try:
                sklearn_model = model._model_impl.sklearn_model
                # Strict schema validation and alignment
                _validate_input_features(input_df, sklearn_model)
                input_df = _align_features(input_df, sklearn_model)
                # Update background cache and decide masker in model space
                _update_background_cache(request.model_name, input_df)

                pre, clf, is_pipe = _get_pipeline_parts(sklearn_model)
                if is_pipe and pre is not None:
                    bg_orig = _get_background_for_model(request.model_name, input_df)
                    bg_df = pd.DataFrame(bg_orig, columns=list(input_df.columns))
                    bg_trans = pre.transform(bg_df)
                    masker = shap.maskers.Independent(bg_trans)
                    model_fn = clf.predict_proba
                else:
                    background = _get_background_for_model(request.model_name, input_df)
                    masker = shap.maskers.Independent(background)
                    model_fn = sklearn_model.predict_proba

                cache_key = f"shap_explainer_{request.model_name}"
                if cache_key not in model_cache:
                    explainer = shap.Explainer(
                        model_fn, masker, algorithm="auto", link="logit"
                    )
                    model_cache[cache_key] = explainer
                    explainer_type = "Explainer(auto/logit)"
                else:
                    explainer = model_cache[cache_key]
                    explainer_type = "Explainer(cache)"

                if is_pipe and pre is not None:
                    X_trans = pre.transform(input_df)
                    sv = explainer(X_trans)
                else:
                    sv = explainer(input_df.values)
                # SHAP values shape: (n_samples, n_features, n_outputs)
                # or (n_samples, n_features) depending on model output
                values = sv.values

                # Choose class index: use predicted class if available
                class_index = 0
                try:
                    if is_pipe:
                        proba = sklearn_model.predict_proba(input_df)[0]
                    else:
                        proba = sklearn_model.predict_proba(input_df)[0]
                    class_index = int(np.argmax(proba))
                except Exception:
                    class_index = 0

                if values.ndim == 3:
                    shap_values = values[0, :, class_index]
                elif values.ndim == 2:
                    shap_values = values[0]
                else:
                    shap_values = values

                # Retry with broader background if all near-zero
                if np.allclose(shap_values, 0, atol=1e-9):
                    if is_pipe and pre is not None:
                        bg_orig = _build_background_from_input(
                            input_df, n=256, scale=0.5
                        )
                        bg_df = pd.DataFrame(bg_orig, columns=list(input_df.columns))
                        bg_trans = pre.transform(bg_df)
                        masker = shap.maskers.Independent(bg_trans)
                        explainer_retry = shap.Explainer(
                            clf.predict_proba, masker, algorithm="auto", link="logit"
                        )
                        X_trans = pre.transform(input_df)
                        sv2 = explainer_retry(X_trans)
                    else:
                        background = _build_background_from_input(
                            input_df, n=256, scale=0.5
                        )
                        masker = shap.maskers.Independent(background)
                        explainer_retry = shap.Explainer(
                            sklearn_model.predict_proba,
                            masker,
                            algorithm="auto",
                            link="logit",
                        )
                        sv2 = explainer_retry(input_df.values)
                    vals2 = sv2.values
                    if vals2.ndim == 3:
                        shap_values = vals2[0, :, class_index]
                    elif vals2.ndim == 2:
                        shap_values = vals2[0]
                    else:
                        shap_values = vals2

                # Calculate feature importance
                feature_importance = {}
                for i, col in enumerate(input_df.columns):
                    shap_val = float(shap_values[i])
                    feat_val = float(input_df[col].values[0])
                    feature_importance[col] = {
                        "shap_value": shap_val,
                        "feature_value": feat_val,
                        "impact": (
                            "increases risk" if shap_val > 0 else "decreases risk"
                        ),
                        "magnitude": abs(shap_val),
                    }

                # Sort by magnitude
                sorted_features = sorted(
                    feature_importance.items(),
                    key=lambda x: x[1]["magnitude"],
                    reverse=True,
                )

                # Handle expected_value (also an array for binary classification)
                base_value = None
                if hasattr(explainer, "expected_value"):
                    ev = explainer.expected_value
                    if isinstance(ev, (list, np.ndarray)):
                        # For binary classification, use positive class
                        base_value = float(ev[1]) if len(ev) > 1 else float(ev[0])
                    else:
                        base_value = float(ev)

                explanation_result = {
                    "method": f"SHAP ({explainer_type})",
                    "feature_importance": dict(sorted_features[:10]),  # Top 10 features
                    "base_value": base_value,
                    "prediction_impact": float(np.sum(shap_values)),
                    "diagnostics": {
                        "sum_abs": float(np.sum(np.abs(shap_values))),
                        "all_zero": bool(np.allclose(shap_values, 0, atol=1e-9)),
                        "background_used": (
                            "cache"
                            if len(BACKGROUND_CACHE.get(request.model_name, [])) >= 64
                            else "jitter"
                        ),
                    },
                    "top_risk_factors": [
                        f[0] for f in sorted_features[:5] if f[1]["shap_value"] > 0
                    ],
                    "top_protective_factors": [
                        f[0] for f in sorted_features[:5] if f[1]["shap_value"] < 0
                    ],
                }

                logger.info("SHAP explanation generated successfully")

            except Exception as e:
                logger.error(f"❌ SHAP explanation failed: {e}", exc_info=True)
                explanation_result = {
                    "error": f"SHAP explanation failed: {str(e)}",
                    "method": "shap",
                    "status": "failed",
                }

        elif request.explanation_type.lower() == "lime":
            try:
                from lime.lime_tabular import LimeTabularExplainer

                # Get sklearn model
                sklearn_model = model._model_impl.sklearn_model

                # Create LIME explainer with synthetic background to avoid zero weights
                x0 = input_df.values[0]
                rng = np.random.default_rng(13)
                scale = np.maximum(np.abs(x0) * 0.05, 0.01)
                lime_background = np.vstack(
                    [x0 + rng.normal(0, scale) for _ in range(200)]
                )

                lime_explainer = LimeTabularExplainer(
                    training_data=lime_background,
                    feature_names=list(input_df.columns),
                    mode="classification",
                )

                # Generate explanation
                exp = lime_explainer.explain_instance(
                    data_row=input_df.values[0],
                    predict_fn=sklearn_model.predict_proba,
                    num_features=min(10, len(input_df.columns)),
                )

                # Extract feature importance
                feature_importance = {}
                for feat, weight in exp.as_list():
                    # Parse feature string (e.g., "age <= 45.0")
                    feat_name = feat.split()[0] if " " in feat else feat
                    if feat_name in input_df.columns:
                        feature_importance[feat_name] = {
                            "lime_weight": float(weight),
                            "feature_value": float(input_df[feat_name].values[0]),
                            "impact": (
                                "increases risk" if weight > 0 else "decreases risk"
                            ),
                            "magnitude": abs(float(weight)),
                            "condition": feat,
                        }

                # Sort by magnitude
                sorted_features = sorted(
                    feature_importance.items(),
                    key=lambda x: x[1]["magnitude"],
                    reverse=True,
                )

                explanation_result = {
                    "method": "LIME (Local Interpretable Model-Agnostic Explanations)",
                    "feature_importance": dict(sorted_features[:10]),
                    "model_score": float(exp.score) if hasattr(exp, "score") else None,
                    "intercept": (
                        float(exp.intercept[1])
                        if hasattr(exp, "intercept") and len(exp.intercept) > 1
                        else None
                    ),
                }

                logger.info("✅ LIME explanation generated successfully")

            except Exception as e:
                logger.error(f"❌ LIME explanation failed: {e}", exc_info=True)
                explanation_result = {
                    "error": f"LIME explanation failed: {str(e)}",
                    "method": "lime",
                    "status": "failed",
                }

        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown explanation type: {request.explanation_type}. "
                    "Use 'shap' or 'lime'"
                ),
            )

        # Add personalization if requested
        personalization_result = None
        if (
            request.include_personalization
            and PERSONALIZATION_AVAILABLE
            and personalization_engine
        ):
            try:
                # TODO: Implement personalization integration
                personalization_result = {
                    "status": "coming_soon",
                    "message": "Personalization integration in progress",
                }
            except Exception as e:
                logger.warning(f"Personalization failed: {e}")

        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000

        return {
            "model_name": request.model_name,
            "explanation_type": request.explanation_type,
            "explanation": explanation_result,
            "personalization": personalization_result,
            "inference_time_ms": round(inference_time, 2),
            "trace_id": trace_id,
            "timestamp": datetime.now().isoformat(),
            "status": "success",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Explanation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@app.post("/personalize")
async def personalize_treatment(request: PersonalizeRequest):
    """
    Generate personalized treatment recommendations

    Finds similar patients and provides cohort-based insights
    """
    trace_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        # Check if personalization is available
        if not PERSONALIZATION_AVAILABLE or personalization_engine is None:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Personalization features not available",
                    "message": "Personalization engine not initialized",
                    "trace_id": trace_id,
                },
            )

        logger.info(f"[{trace_id}] Finding similar patients for {request.model_name}")

        # Prepare patient data summary (reserved for future use)

        # TODO: Full personalization integration
        # For now, return structured placeholder
        personalization_result = {
            "status": "in_development",
            "message": "Personalization engine is being integrated",
            "features_available": {
                "patient_similarity": "coming_soon",
                "cohort_identification": "coming_soon",
                "treatment_recommendations": "coming_soon",
                "outcome_prediction": "coming_soon",
            },
            "patient_features_received": len(request.patient_data.features),
            "k_neighbors_requested": request.k_neighbors,
        }

        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000

        return {
            "model_name": request.model_name,
            "personalization": personalization_result,
            "inference_time_ms": round(inference_time, 2),
            "trace_id": trace_id,
            "timestamp": datetime.now().isoformat(),
            "status": "partial",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Personalization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Personalization failed: {str(e)}")


@app.get("/metrics")
async def get_metrics():
    """Get API performance metrics"""
    return {
        "uptime_seconds": time.time() - START_TIME,
        "models_cached": len(MODEL_CACHE),
        "cache_details": {
            name: {
                "loaded_at": data["loaded_at"],
                "load_time_seconds": data["load_time_seconds"],
            }
            for name, data in MODEL_CACHE.items()
        },
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested endpoint does not exist",
            "path": str(request.url),
        },
    )


@app.exception_handler(500)
async def server_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "trace_id": str(uuid.uuid4()),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
