from fastapi import FastAPI, status, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any
import numpy as np
import logging
import os
from datetime import datetime

from .predict import predict_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Iris Classification API",
    description="API for classifying iris flowers using machine learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class IrisData(BaseModel):
    """
    Input data model for iris flower classification.
    
    Attributes:
        sepal_length (float): Sepal length in cm (0 < value ≤ 10)
        sepal_width (float): Sepal width in cm (0 < value ≤ 10)
        petal_length (float): Petal length in cm (0 < value ≤ 10)
        petal_width (float): Petal width in cm (0 < value ≤ 10)
    """
    sepal_length: float = Field(..., 
        gt=0, 
        le=10, 
        description="Sepal length in centimeters (0 < value ≤ 10)",
        example=5.1
    )
    sepal_width: float = Field(..., 
        gt=0, 
        le=10, 
        description="Sepal width in centimeters (0 < value ≤ 10)",
        example=3.5
    )
    petal_length: float = Field(..., 
        gt=0, 
        le=10, 
        description="Petal length in centimeters (0 < value ≤ 10)",
        example=1.4
    )
    petal_width: float = Field(..., 
        gt=0, 
        le=10, 
        description="Petal width in centimeters (0 < value ≤ 10)",
        example=0.2
    )

    @validator('*')
    def check_positive(cls, v):
        if v <= 0:
            raise ValueError('All measurements must be positive')
        return v

class PredictionResponse(BaseModel):
    prediction: str
    class_id: int
    probabilities: List[Dict[str, float]]
    model_version: str
    class_names: List[str]

# Minimal root endpoint
@app.get("/")
async def root():
    return {
        "message": "Iris Classification API is running",
        "docs": "/docs",
        "health": "/health"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post(
    "/predict", 
    response_model=PredictionResponse, 
    status_code=status.HTTP_200_OK,
    summary="Predict Iris Species",
    description="""
    Predict the species of an iris flower based on its measurements.
    
    This endpoint accepts four measurements of an iris flower and returns
    the predicted species along with probability scores for each class.
    
    ### Example Request
    ```json
    {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    ```
    """,
    response_description="Prediction result with probabilities"
)
async def predict_iris(iris_features: IrisData):
    """
    Predict the species of an iris flower based on its measurements.
    
    Args:
        iris_features (IrisData): The input features for prediction
        request (Request): The incoming request object
        
    Returns:
        dict: Prediction results including class, probabilities, and model version
        
    Raises:
        HTTPException: If there's an error during prediction
    """
    try:
        logger.info(f"Received prediction request: {iris_features.dict()}")
        
        # Prepare features in the correct order (matching training data)
        features = [[
            iris_features.sepal_length,
            iris_features.sepal_width,
            iris_features.petal_length,
            iris_features.petal_width
        ]]

        # Get prediction and metadata
        predictions, metadata = predict_data(np.array(features))
        
        # Format response
        prediction = int(predictions[0])
        class_name = metadata['class_names'][prediction]
        
        response = {
            "prediction": class_name,
            "class_id": prediction,
            "probabilities": metadata['probabilities'],
            "model_version": metadata['model_version'],
            "class_names": metadata['class_names']
        }
        
        logger.info(f"Prediction successful: {response}")
        return response
        
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(ve)}"
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during prediction: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
