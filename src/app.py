from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
from src.inference import predict_image

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Cats vs Dogs Inference API",
    description="Simple binary image classification API",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                    # For testing; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", tags=["health"])
async def health_check():
    """Health endpoint to check if service is running"""
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", tags=["prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Upload an image (cat or dog) and get prediction
    """
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File is not an image")
        
        contents = await file.read()
        
        result = predict_image(contents)
        
        logger.info(f"Prediction: {result['predicted_class']} | Confidence: {result['confidence']}")
        
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))