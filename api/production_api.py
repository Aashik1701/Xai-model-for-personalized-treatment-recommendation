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
    logger.info(f"   - Predictions: ✅ Enabled")
    logger.info(
        f"   - Explainability: {'✅ Enabled' if explainability_toolkit else '❌ Disabled'}"
    )
    logger.info(
        f"   - Personalization: {'✅ Enabled' if personalization_engine else '❌ Disabled'}"
    )
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


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction for a single patient"""
    trace_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        # Load model
        model_data = load_model(request.model_name)
        model = model_data["model"]

        # Prepare input data
        input_df = pd.DataFrame([request.patient_data.features])

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

                # Create SHAP explainer (cached per model)
                cache_key = f"shap_{request.model_name}"
                if cache_key not in model_cache:
                    # Get the actual sklearn model from the MLflow wrapper
                    sklearn_model = model._model_impl.sklearn_model
                    explainer = shap.TreeExplainer(sklearn_model)
                    model_cache[cache_key] = explainer
                else:
                    explainer = model_cache[cache_key]

                # Calculate SHAP values
                shap_values_raw = explainer.shap_values(input_df)

                # Handle multi-output (binary classification returns list of 2)
                if isinstance(shap_values_raw, list):
                    # For binary classification, use positive class (index 1)
                    shap_values = (
                        shap_values_raw[1]
                        if len(shap_values_raw) > 1
                        else shap_values_raw[0]
                    )
                else:
                    shap_values = shap_values_raw

                # Convert to numpy array and ensure proper shape
                shap_values = np.array(shap_values)

                # For single sample, should be shape (n_features,)
                # If shape is (1, n_features), squeeze first dimension
                if len(shap_values.shape) > 1 and shap_values.shape[0] == 1:
                    shap_values = shap_values[0]

                # For binary classification: (n_features, 2) -> take class 1
                # For multiclass: (n_features, n_classes) -> take predicted class
                if len(shap_values.shape) == 2 and shap_values.shape[0] == len(
                    input_df.columns
                ):
                    # Shape is (n_features, n_classes)
                    # Use positive class (index 1) for binary, or max class
                    if shap_values.shape[1] == 2:
                        shap_values = shap_values[:, 1]  # Positive class
                    else:
                        # For multiclass, use the predicted class
                        pred_class = int(prediction[0])
                        shap_values = shap_values[:, pred_class]

                # Verify final shape
                logger.info(
                    f"Final SHAP shape: {shap_values.shape}, "
                    f"Features: {len(input_df.columns)}"
                )

                if len(shap_values) != len(input_df.columns):
                    raise ValueError(
                        f"SHAP values length {len(shap_values)} != "
                        f"features {len(input_df.columns)}"
                    )

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

                # Sort by magnitude
                sorted_features = sorted(
                    feature_importance.items(),
                    key=lambda x: x[1]["magnitude"],
                    reverse=True,
                )

                explanation_result = {
                    "method": "SHAP",
                    "top_features": dict(sorted_features[:5]),
                    "total_features": len(feature_importance),
                }

                logger.info(f"✅ SHAP explanation generated")

            except Exception as e:
                logger.warning(f"⚠️ SHAP explanation failed: {e}")
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

                # Simple cohort assignment based on risk score
                risk_score = float(confidence)  # Use model confidence as risk

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

                logger.info(f"✅ Personalization generated - Cohort: {cohort}")

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
        except:
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
                    "message": "SHAP/LIME libraries not installed or toolkit initialization failed",
                    "trace_id": trace_id,
                    "workaround": "Install with: pip install shap lime",
                },
            )

        logger.info(
            f"[{trace_id}] Generating {request.explanation_type} explanation for {request.model_name}"
        )

        # Load model from MLflow
        model_uri = f"models:/{request.model_name}/Production"
        try:
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"✅ Loaded model: {request.model_name}")
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.model_name}' not found in Production: {str(e)}",
            )

        # Prepare input data
        input_df = pd.DataFrame([request.patient_data.features])

        # Generate explanation based on type
        explanation_result = {}

        if request.explanation_type.lower() == "shap":
            try:
                # Create SHAP explainer (cached per model)
                cache_key = f"shap_{request.model_name}"
                if cache_key not in model_cache:
                    # Get the actual sklearn model from the MLflow wrapper
                    sklearn_model = model._model_impl.sklearn_model
                    explainer = shap.TreeExplainer(sklearn_model)
                    model_cache[cache_key] = explainer
                    logger.info(
                        f"✅ Created new SHAP explainer for {request.model_name}"
                    )
                else:
                    explainer = model_cache[cache_key]

                # Calculate SHAP values
                shap_values_raw = explainer.shap_values(input_df)

                # Handle multi-output (binary classification returns list of 2)
                if isinstance(shap_values_raw, list):
                    # For binary classification, use positive class (index 1)
                    shap_values = (
                        shap_values_raw[1]
                        if len(shap_values_raw) > 1
                        else shap_values_raw[0]
                    )
                else:
                    shap_values = shap_values_raw

                # Convert to numpy array
                shap_values = np.array(shap_values)

                # For single sample: should be (n_features,)
                # If shape is (1, n_features), squeeze first dimension
                if len(shap_values.shape) > 1 and shap_values.shape[0] == 1:
                    shap_values = shap_values[0]

                # For binary classification: (n_features, 2) -> take class 1
                if len(shap_values.shape) == 2 and shap_values.shape[0] == len(
                    input_df.columns
                ):
                    # Shape is (n_features, n_classes)
                    if shap_values.shape[1] == 2:
                        shap_values = shap_values[:, 1]  # Positive class
                    else:
                        # For multiclass, use max probability class
                        shap_values = shap_values[:, 0]

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
                    "method": "SHAP (TreeExplainer)",
                    "feature_importance": dict(sorted_features[:10]),  # Top 10 features
                    "base_value": base_value,
                    "prediction_impact": float(np.sum(shap_values)),
                    "top_risk_factors": [
                        f[0] for f in sorted_features[:5] if f[1]["shap_value"] > 0
                    ],
                    "top_protective_factors": [
                        f[0] for f in sorted_features[:5] if f[1]["shap_value"] < 0
                    ],
                }

                logger.info(f"✅ SHAP explanation generated successfully")

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

                # Create LIME explainer
                # Note: LIME needs training data statistics, using simple approach
                lime_explainer = LimeTabularExplainer(
                    training_data=input_df.values,
                    feature_names=list(input_df.columns),
                    mode="classification",
                )

                # Generate explanation
                exp = lime_explainer.explain_instance(
                    data_row=input_df.values[0],
                    predict_fn=sklearn_model.predict_proba,
                    num_features=10,
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
                detail=f"Unknown explanation type: {request.explanation_type}. Use 'shap' or 'lime'",
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

        # Prepare patient data
        input_df = pd.DataFrame([request.patient_data.features])

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
