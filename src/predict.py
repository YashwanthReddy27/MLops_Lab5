import joblib
import numpy as np
import os
from typing import Tuple, Dict, Any

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')

def get_latest_model() -> Tuple[str, str]:
    """
    Get the path to the latest trained model.
    
    Returns:
        tuple: (model_path, model_version)
    """
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Look for both file patterns
    model_files = []
    
    # Pattern 1: iris_model_vX_Y_Z.pkl (versioned)
    versioned_models = [f for f in os.listdir(MODEL_DIR) 
                       if f.startswith('iris_model_v') and f.endswith('.pkl')]
    
    # Pattern 2: iris_model.pkl (legacy)
    legacy_models = [f for f in os.listdir(MODEL_DIR) 
                    if f == 'iris_model.pkl']
    
    model_files = versioned_models + legacy_models
    
    if not model_files:
        raise FileNotFoundError("No trained model found in the model directory")
    
    # If we have versioned models, get the latest one
    if versioned_models:
        def get_version(f):
            return tuple(map(int, f.split('_v')[1].split('.pkl')[0].split('_')))
        
        latest_model = sorted(versioned_models, key=get_version, reverse=True)[0]
        version = latest_model.split('_v')[1].split('.pkl')[0].replace('_', '.')
    else:
        # Use the legacy model file
        latest_model = 'iris_model.pkl'
        version = '1.0.0'  # Default version for legacy models
    
    return os.path.join(MODEL_DIR, latest_model), version

def predict_data(X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Predict the class labels and probabilities for the input data.
    
    Args:
        X: Input data for which predictions are to be made.
        
    Returns:
        A tuple containing:
            - predictions: numpy array of predicted class labels
            - prediction_metadata: dict containing probabilities, model version, etc.
    """
    try:
        model_path, model_version = get_latest_model()
        model = joblib.load(model_path)
        
        # Get predictions and probabilities
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Get class names from the model or use default
        if hasattr(model, 'classes_'):
            class_names = [str(cls) for cls in model.classes_]
        else:
            class_names = [f"class_{i}" for i in range(probabilities.shape[1])]
        
        # Format probabilities for each prediction
        prob_list = []
        for prob in probabilities:
            prob_list.append(dict(zip(class_names, [float(round(p, 4)) for p in prob])))
        
        return predictions, {
            'model_version': model_version,
            'probabilities': prob_list,
            'class_names': class_names
        }
        
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")
