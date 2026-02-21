import os
import numpy as np
from PIL import Image
import io
import logging
import tensorflow as tf
import mlflow
import mlflow.pyfunc

# load the model from the local path
MODEL_PATH = "models/baseline_model.keras"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

# Global model
model = None

logger = logging.getLogger(__name__)

def load_model():
    global model
    if model is not None:
        return model

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model = mlflow.pyfunc.load_model("models:/cats-vs-dogs-baseline/1")
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        model = None

    return model

def predict_image(image_bytes: bytes):
    # Load model if not already loaded
    mdl = load_model()
    
    # Preprocess image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 224, 224, 3)
    
    # Predict
    pred = mdl.predict(img_array)[0][0]
    
    # Binary classification
    label = "dog" if pred > 0.5 else "cat"
    confidence = pred if pred > 0.5 else 1 - pred
    confidence = float(confidence) * 100  # as percentage
    
    return {
        "predicted_class": label,
        "confidence": f"{confidence:.2f}%",
        "raw_probability_dog": float(pred)
    }