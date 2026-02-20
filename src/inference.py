import os
import numpy as np
from PIL import Image
import io
import dvc.api
import logging
import tensorflow as tf

# load the model from the local path (later change to MLflow artifact)
MODEL_PATH = "models/baseline_model.keras"

# Global model (loaded once at startup - good for FastAPI)
model = None

logger = logging.getLogger(__name__)

def load_model():
    global model
    if model is not None:
        return model

    if not os.path.exists(MODEL_PATH):
        logger.info("Model not found locally → pulling from DVC (Backblaze B2)")
        try:
            dvc.api.get("models/baseline_model.keras", out=MODEL_PATH)
            logger.info("Model successfully pulled from B2")
        except Exception as e:
            logger.error(f"Failed to pull model: {str(e)}")
            raise RuntimeError("Model unavailable - service in health-only mode")

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
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
    pred = mdl.predict(img_array, verbose=0)[0][0]
    
    # Binary classification
    label = "dog" if pred > 0.5 else "cat"
    confidence = pred if pred > 0.5 else 1 - pred
    confidence = float(confidence) * 100  # as percentage
    
    return {
        "predicted_class": label,
        "confidence": f"{confidence:.2f}%",
        "raw_probability_dog": float(pred)
    }