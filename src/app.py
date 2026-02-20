from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.inference import predict_image, load_model
import logging
import time
from collections import Counter
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Cats vs Dogs Inference API",
    description="Binary image classification for pet adoption",
    version="0.1.0"
)

# CORS (for Swagger UI if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory counters (for demo - reset on restart)
request_counter = Counter()
latency_histogram = []  # list of latencies

@app.get("/health", tags=["health"])
async def health_check():
    mdl = load_model()
    status = "healthy" if mdl is not None else "healthy (no model loaded)"
    logger.info("Health check called")
    return {"status": status, "request_count": request_counter["predict"]}

@app.post("/predict", tags=["prediction"])
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    request_counter["predict"] += 1
    
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(400, "File must be an image")
        
        contents = await file.read()
        result = predict_image(contents)
        
        latency = time.time() - start_time
        latency_histogram.append(latency)
        
        logger.info(
            f"Prediction success | class: {result['predicted_class']} | "
            f"conf: {result['confidence']} | latency: {latency:.3f}s | "
            f"total requests: {request_counter['predict']}"
        )
        
        return result
    
    except Exception as e:
        latency = time.time() - start_time
        logger.error(f"Prediction failed: {str(e)} | latency: {latency:.3f}s")
        raise HTTPException(500, str(e))

@app.get("/metrics", tags=["monitoring"])
async def get_metrics():
    """Simple endpoint to show request count and avg latency"""
    if not latency_histogram:
        avg_latency = 0.0
    else:
        avg_latency = sum(latency_histogram) / len(latency_histogram)
    
    return {
        "total_requests": request_counter["predict"],
        "avg_latency_seconds": round(avg_latency, 3),
        "log_file": "inference.log (contains detailed request logs)"
    }