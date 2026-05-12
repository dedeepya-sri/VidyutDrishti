"""
VidyutDrishti FastAPI Backend
Production-ready API for renewable energy forecasting
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
from contextlib import asynccontextmanager

# Import models
from models.lightgbm.quantile_regression import QuantileLightGBM
from models.ensemble.hybrid_ensemble import HybridEnsemble
from models.evaluation.model_evaluation import RenewableEnergyEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instances
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup"""
    logger.info("Loading models...")

    try:
        # Load LightGBM model
        lgbm_model = QuantileLightGBM()
        lgbm_model.load_model("./models/lightgbm/quantile_model.pkl")
        models['lightgbm'] = lgbm_model
        logger.info("LightGBM model loaded")
    except Exception as e:
        logger.error(f"Failed to load LightGBM model: {e}")

    try:
        # Load ensemble model
        ensemble_model = HybridEnsemble()
        ensemble_model.load_ensemble("./models/ensemble/hybrid_model.pkl")
        models['ensemble'] = ensemble_model
        logger.info("Ensemble model loaded")
    except Exception as e:
        logger.error(f"Failed to load ensemble model: {e}")

    # Load evaluator
    try:
        evaluator = RenewableEnergyEvaluator()
        models['evaluator'] = evaluator
        logger.info("Evaluator loaded")
    except Exception as e:
        logger.error(f"Failed to load evaluator: {e}")

    yield

    # Cleanup on shutdown
    models.clear()
    logger.info("Models unloaded")

# Create FastAPI app
app = FastAPI(
    title="VidyutDrishti API",
    description="Production-ready renewable energy forecasting system for Karnataka SLDC",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ForecastRequest(BaseModel):
    """Single forecast request"""
    temperature_2m: float = Field(..., description="Temperature at 2m (°C)")
    wind_speed: float = Field(..., description="Wind speed (m/s)")
    surface_solar_radiation_downwards: float = Field(..., description="Solar irradiance (W/m²)")
    total_cloud_cover: float = Field(..., description="Cloud cover (0-100%)")
    total_precipitation: float = Field(..., description="Precipitation (mm)")
    hour: int = Field(..., ge=0, le=23, description="Hour of day")
    month: int = Field(..., ge=1, le=12, description="Month")
    plant_type: str = Field(..., description="Plant type (solar/wind)")
    model: str = Field("ensemble", description="Model to use (lightgbm/ensemble)")

class BatchForecastRequest(BaseModel):
    """Batch forecast request"""
    forecasts: List[ForecastRequest] = Field(..., description="List of forecast requests")
    model: str = Field("ensemble", description="Model to use")

class ForecastResponse(BaseModel):
    """Forecast response with uncertainty"""
    timestamp: Optional[datetime] = Field(None, description="Forecast timestamp")
    P10: float = Field(..., description="10th percentile forecast (MW)")
    P50: float = Field(..., description="50th percentile forecast (MW)")
    P90: float = Field(..., description="90th percentile forecast (MW)")
    uncertainty_range: float = Field(..., description="Uncertainty range P90-P10 (MW)")
    confidence_level: str = Field(..., description="Confidence assessment")

class BatchForecastResponse(BaseModel):
    """Batch forecast response"""
    forecasts: List[ForecastResponse]
    model_used: str
    processing_time: float
    total_forecasts: int

class ExplainRequest(BaseModel):
    """Explainability request"""
    forecast_request: ForecastRequest
    model: str = Field("ensemble", description="Model to explain")

class ExplainResponse(BaseModel):
    """Explainability response"""
    forecast: ForecastResponse
    feature_importance: Dict[str, float]
    key_drivers: List[str]
    shap_values: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    models_loaded: List[str]
    version: str

# Utility functions
def get_model(model_name: str):
    """Get model instance with error handling"""
    if model_name not in models:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not available")
    return models[model_name]

def assess_confidence(uncertainty_range: float, capacity: float = 100) -> str:
    """Assess forecast confidence based on uncertainty"""
    uncertainty_ratio = uncertainty_range / capacity

    if uncertainty_ratio < 0.1:
        return "High"
    elif uncertainty_ratio < 0.2:
        return "Medium"
    else:
        return "Low"

def preprocess_request(request: ForecastRequest) -> Dict:
    """Preprocess forecast request into model features"""
    # Add derived features
    features = request.dict()

    # Cyclical encoding
    features['hour_sin'] = np.sin(2 * np.pi * request.hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * request.hour / 24)
    features['month_sin'] = np.sin(2 * np.pi * request.month / 12)
    features['month_cos'] = np.cos(2 * np.pi * request.month / 12)

    # Remove non-feature fields
    features.pop('model', None)

    return features

# API endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "VidyutDrishti API - Karnataka SLDC Renewable Energy Forecasting",
        "version": "1.0.0",
        "documentation": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    loaded_models = list(models.keys())

    return HealthResponse(
        status="healthy" if loaded_models else "degraded",
        timestamp=datetime.now(),
        models_loaded=loaded_models,
        version="1.0.0"
    )

@app.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest):
    """Single forecast with uncertainty"""
    start_time = time.time()

    try:
        # Get model
        model = get_model(request.model)

        # Preprocess request
        features = preprocess_request(request)

        # Generate forecast
        prediction = model.predict_single(features)

        # Calculate uncertainty metrics
        uncertainty_range = prediction.get('P90', 0) - prediction.get('P10', 0)

        # Assess confidence (simplified - would use capacity from plant data)
        confidence = assess_confidence(uncertainty_range)

        response = ForecastResponse(
            P10=round(prediction.get('P10', 0), 2),
            P50=round(prediction.get('P50', 0), 2),
            P90=round(prediction.get('P90', 0), 2),
            uncertainty_range=round(uncertainty_range, 2),
            confidence_level=confidence
        )

        logger.info(".2f")

        return response

    except Exception as e:
        logger.error(f"Forecast error: {e}")
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")

@app.post("/batch_forecast", response_model=BatchForecastResponse)
async def batch_forecast(request: BatchForecastRequest):
    """Batch forecast for multiple inputs"""
    start_time = time.time()

    try:
        # Get model
        model = get_model(request.model)

        forecasts = []

        for forecast_request in request.forecasts:
            # Preprocess request
            features = preprocess_request(forecast_request)

            # Generate forecast
            prediction = model.predict_single(features)

            # Calculate uncertainty
            uncertainty_range = prediction.get('P90', 0) - prediction.get('P10', 0)
            confidence = assess_confidence(uncertainty_range)

            forecast_response = ForecastResponse(
                P10=round(prediction.get('P10', 0), 2),
                P50=round(prediction.get('P50', 0), 2),
                P90=round(prediction.get('P90', 0), 2),
                uncertainty_range=round(uncertainty_range, 2),
                confidence_level=confidence
            )

            forecasts.append(forecast_response)

        processing_time = time.time() - start_time

        response = BatchForecastResponse(
            forecasts=forecasts,
            model_used=request.model,
            processing_time=round(processing_time, 3),
            total_forecasts=len(forecasts)
        )

        logger.info(f"Batch forecast completed: {len(forecasts)} forecasts in {processing_time:.2f}s")

        return response

    except Exception as e:
        logger.error(f"Batch forecast error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch forecast failed: {str(e)}")

@app.post("/explain", response_model=ExplainResponse)
async def explain_forecast(request: ExplainRequest):
    """Explain forecast with feature importance"""
    try:
        # Get model
        model = get_model(request.model)

        # Preprocess request
        features = preprocess_request(request.forecast_request)

        # Generate forecast
        prediction = model.predict_single(features)

        # Get feature importance (if available)
        feature_importance = {}
        key_drivers = []

        if hasattr(model, 'get_feature_importance'):
            try:
                importance_df = model.get_feature_importance()
                feature_importance = dict(zip(
                    importance_df['feature'].head(10),
                    importance_df['importance'].head(10)
                ))

                # Get top drivers
                key_drivers = importance_df['feature'].head(5).tolist()

            except Exception as e:
                logger.warning(f"Could not get feature importance: {e}")

        # Calculate uncertainty
        uncertainty_range = prediction.get('P90', 0) - prediction.get('P10', 0)
        confidence = assess_confidence(uncertainty_range)

        forecast_response = ForecastResponse(
            P10=round(prediction.get('P10', 0), 2),
            P50=round(prediction.get('P50', 0), 2),
            P90=round(prediction.get('P90', 0), 2),
            uncertainty_range=round(uncertainty_range, 2),
            confidence_level=confidence
        )

        return ExplainResponse(
            forecast=forecast_response,
            feature_importance=feature_importance,
            key_drivers=key_drivers
        )

    except Exception as e:
        logger.error(f"Explain error: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Get model performance metrics"""
    try:
        evaluator = models.get('evaluator')
        if not evaluator:
            raise HTTPException(status_code=503, detail="Evaluator not available")

        # Get comparison data
        comparison = evaluator.compare_models()

        return {
            "metrics": comparison.to_dict(),
            "timestamp": datetime.now(),
            "models_evaluated": list(comparison.index)
        }

    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP error: {exc.detail}")
    return {"error": exc.detail, "status_code": exc.status_code}

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {exc}")
    return {"error": "Internal server error", "status_code": 500}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)