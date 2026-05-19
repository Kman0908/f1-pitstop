import os
import logging
from typing import Dict
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="F1-PitStop Prediction API",
    description="Predict pit stop timing for F1 drivers",
    version="1.0.0"
)

# Add CORS middleware for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import custom modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.utils import load_obj
    from src.components.data_transformation import DataTransformation
    from src.logger import logging as custom_logging
except ImportError as e:
    logger.warning(f"Could not import src modules: {e}")


# Pydantic models for request/response
class PredictionInput(BaseModel):
    Driver: str = Field(..., description="Driver name (e.g., 'VER', 'HAM', 'ALB')")
    Compound: str = Field(..., description="Tire compound ('SOFT', 'MEDIUM', 'HARD')")
    Race: str = Field(..., description="Race name (e.g., 'Monaco Grand Prix')")
    Year: int = Field(..., ge=2015, le=2024, description="Race year")
    PitStop: int = Field(..., ge=0, le=3, description="Pit stop number (0-3)")
    LapNumber: int = Field(..., ge=1, description="Current lap number")
    Stint: int = Field(..., ge=1, le=5, description="Current stint number")
    TyreLife: float = Field(..., ge=0, description="Tyre life remaining (laps)")
    Position: int = Field(..., ge=1, le=20, description="Current position")
    LapTime_s: float = Field(..., gt=0, description="Current lap time in seconds")   
    LapTime_Delta: float = Field(..., description="Lap time delta vs best lap")
    Cumulative_Degradation: float = Field(..., description="Cumulative tire degradation")
    RaceProgress: float = Field(..., ge=0, le=1, description="Race progress (0-1)")
    Position_Change: float = Field(..., description="Change in position")


class PredictionOutput(BaseModel):
    prediction: int
    prediction_label: str
    confidence_message: str
    input_summary: Dict


class ErrorResponse(BaseModel):
    detail: str
    error_type: str


# Global variables for model and preprocessor
model = None
preprocessor = None
data_transformer = None


def load_model_and_preprocessor():
    """Load model and preprocessor on startup"""
    global model, preprocessor, data_transformer
    
    try:
        model_path = os.path.join(os.getcwd(), 'artifacts', 'objects', 'model.pkl')
        preprocessor_path = os.path.join(os.getcwd(), 'artifacts', 'objects', 'preprocessor.pkl')
        
        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Loading preprocessor from: {preprocessor_path}")
        
        model = load_obj(model_path)
        preprocessor = load_obj(preprocessor_path)
        data_transformer = DataTransformation()
        
        logger.info("✅ Model and preprocessor loaded successfully")
    except FileNotFoundError as e:
        logger.error(f"❌ Model or preprocessor file not found: {e}")
        raise RuntimeError(f"Failed to load model: {e}")
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model_and_preprocessor()


@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "message": "F1-PitStop Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "docs": "/docs (GET)"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None or preprocessor is None:
        return {
            "status": "unhealthy",
            "message": "Model or preprocessor not loaded"
        }
    return {
        "status": "healthy",
        "message": "API is running and model is loaded"
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """
    Make a prediction for pit stop timing
    
    Returns:
        - prediction: 1 if pit stop should happen next lap, 0 otherwise
        - prediction_label: Human-readable prediction
        - confidence_message: Explanation of the prediction
    """
    try:
        logger.info(f"Received prediction request: {input_data}")
        
        # Check if model is loaded
        if model is None or preprocessor is None:
            logger.error("Model or preprocessor not loaded")
            raise HTTPException(
                status_code=503,
                detail="Model not available. Please try again later."
            )
        
        # Convert input to dictionary
        input_dict = input_data.dict()

        # Rename the field to match preprocessor
        if 'LapTime_s' in input_dict:
            input_dict['LapTime (s)'] = input_dict.pop('LapTime_s')

    
        # Create DataFrame from input
        df = pd.DataFrame([input_dict])
        
        logger.info(f"Input data shape: {df.shape}")
        logger.info(f"Input columns: {df.columns.tolist()}")
        
        # Apply feature engineering (same as training)
        df_engineered = data_transformer._engineer_features(df)
        logger.info(f"After feature engineering shape: {df_engineered.shape}")
        
        # Transform features using preprocessor
        features_scaled = preprocessor.transform(df_engineered)
        logger.info(f"Features shape after preprocessing: {features_scaled.shape}")
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        
        logger.info(f"Prediction: {prediction}, Probabilities: {prediction_proba}")
        
        # Prepare response
        prediction_label = "PIT NEXT LAP" if prediction == 1 else "NO PIT"
        confidence = max(prediction_proba) * 100
        confidence_message = f"Model is {confidence:.1f}% confident in this prediction"
        
        return PredictionOutput(
            prediction=int(prediction),
            prediction_label=prediction_label,
            confidence_message=confidence_message,
            input_summary={
                "driver": input_data.Driver,
                "race": input_data.Race,
                "year": input_data.Year,
                "position": input_data.Position,
                "lap_number": input_data.LapNumber
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error during prediction: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict-batch")
async def predict_batch(inputs: list[PredictionInput]):
    """
    Make predictions for multiple inputs
    """
    try:
        if len(inputs) > 100:
            raise HTTPException(
                status_code=400,
                detail="Maximum 100 predictions per request"
            )
        
        results = []
        for input_data in inputs:
            result = await predict(input_data)
            results.append(result)
        
        return {
            "count": len(results),
            "predictions": results
        }
    
    except Exception as e:
        logger.exception(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Batch prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        "backend_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
