"""
FastAPI Application for Credit Risk Prediction
Provides REST API endpoints for fairness-aware credit risk scoring
Includes health checks, predictions, and model metrics endpoints
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from datetime import datetime

from api.schemas.prediction import (
    CreditApplicationRequest, 
    CreditPredictionResponse,
    HealthResponse,
    ModelMetricsResponse
)
from api.utils.model_loader import ModelLoader

# Configure logging for application monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI application with metadata
app = FastAPI(
    title="Credit Risk Scoring API",
    description="Fairness-aware AutoML API for credit risk prediction",
    version="1.0.0",
    docs_url="/docs",      # Swagger UI documentation
    redoc_url="/redoc"     # ReDoc documentation
)

# Configure CORS to allow cross-origin requests
# In production, replace "*" with specific allowed origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # Allow all origins (modify for production)
    allow_credentials=True,
    allow_methods=["*"],           # Allow all HTTP methods
    allow_headers=["*"],           # Allow all headers
)

# Global model loader instance
model_loader = None


@app.on_event("startup")
async def startup_event():
    """
    Startup event handler - loads all ML artifacts when API starts
    Loads model, preprocessor, threshold optimizer, and metadata
    """
    global model_loader
    try:
        model_loader = ModelLoader(artifacts_path="artifacts")
        model_loader.load_all()
        logger.info("API started successfully - all models loaded")
    except Exception as e:
        logger.error(f"Failed to load models on startup: {e}")
        raise


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint providing API information and navigation
    
    Returns:
        Basic API information and links to documentation
    """
    return {
        "message": "Credit Risk Scoring API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint for monitoring API status
    Verifies that all required components are loaded
    
    Returns:
        HealthResponse with status of each component
    """
    return HealthResponse(
        status="healthy",
        model_loaded=model_loader is not None and model_loader.model is not None,
        preprocessor_loaded=model_loader is not None and model_loader.preprocessor is not None,
        fairness_module_loaded=model_loader is not None and model_loader.threshold_optimizer is not None,
        version="1.0.0"
    )


@app.post("/predict", response_model=CreditPredictionResponse, tags=["Prediction"])
async def predict_credit_risk(request: CreditApplicationRequest):
    """
    Predict credit risk for a loan application
    Applies fairness-aware post-processing if gender information is provided
    
    Args:
        request: Credit application data including applicant features
        
    Returns:
        CreditPredictionResponse with prediction, probabilities, and risk level
        
    Raises:
        HTTPException: If prediction fails due to invalid input or processing error
    """
    try:
        # Convert Pydantic model to dictionary for processing
        input_data = request.model_dump()
        
        # Make prediction with fairness adjustment enabled
        result = model_loader.predict(input_data, apply_fairness=True)
        
        # Add model version to response for tracking
        result["model_version"] = "1.0.0"
        
        # Log prediction for monitoring
        logger.info(
            f"Prediction made: {result['prediction_label']} "
            f"(probability: {result['probability_default']:.3f})"
        )
        
        return CreditPredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/metrics", response_model=ModelMetricsResponse, tags=["Model Info"])
async def get_model_metrics():
    """
    Retrieve model performance and fairness metrics
    Provides comprehensive evaluation metrics for the deployed model
    
    Returns:
        ModelMetricsResponse with performance and fairness metrics
        
    Raises:
        HTTPException: If metrics are not available or retrieval fails
    """
    try:
        # Retrieve model metadata from loader
        metadata = model_loader.get_metrics()
        
        if metadata is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model metrics not available"
            )
        
        # Extract performance and fairness metrics
        performance = metadata['performance']
        fairness = metadata['fairness']
        
        # Evaluate fairness compliance based on thresholds
        # Disparate impact should be between 0.8 and 1.25
        di_pass = 0.8 <= fairness['disparate_impact'] <= 1.25
        # Statistical parity difference should be within ±0.1
        sp_pass = abs(fairness['statistical_parity_difference']) <= 0.1
        fairness_compliant = di_pass and sp_pass
        
        return ModelMetricsResponse(
            roc_auc=performance['roc_auc'],
            balanced_accuracy=performance['balanced_accuracy'],
            f1_score=performance['f1_score'],
            precision=performance['precision'],
            recall=performance['recall'],
            disparate_impact=fairness['disparate_impact'],
            statistical_parity_difference=fairness['statistical_parity_difference'],
            equal_opportunity_difference=fairness['equal_opportunity_difference'],
            fairness_compliant=fairness_compliant
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
    except Exception as e:
        logger.error(f"Error fetching metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch metrics: {str(e)}"
        )


@app.get("/model-info", tags=["Model Info"])
async def get_model_info():
    """
    Get detailed information about the deployed model
    Provides model architecture, preprocessing, and fairness mitigation details
    
    Returns:
        Dictionary with comprehensive model information
    """
    return {
        "model_type": "Random Forest (AutoML Optimized)",
        "fairness_aware": True,
        "features": len(model_loader.feature_columns) if model_loader else 0,
        "preprocessing": "StandardScaler + Median Imputation",
        "class_imbalance_handling": "class_weight='balanced'",
        "fairness_mitigation": "AIF360 Reweighting + Threshold Optimization",
        "trained_on": "German Credit Dataset",
        "last_updated": datetime.now().isoformat()
    }


# Custom error handlers for better error responses

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """
    Custom handler for 404 Not Found errors
    Provides helpful error message with requested path
    """
    return JSONResponse(
        status_code=404,
        content={
            "message": "Endpoint not found", 
            "path": str(request.url)
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """
    Custom handler for 500 Internal Server errors
    Logs error and returns safe error message
    """
    return JSONResponse(
        status_code=500,
        content={
            "message": "Internal server error", 
            "detail": str(exc)
        }
    )