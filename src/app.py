from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.inference import predict_image, load_model
import logging
import time
from collections import Counter

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
    title="MLOPS Binary Classification Inference API",
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

request_count = 0
latency_sum = 0.0
latency_count = 0

@app.get("/health")
async def health():
    mdl = load_model()
    status = "healthy" if mdl is not None else "healthy (model not loaded)"
    return {
        "status": status,
        "request_count": request_count,
        "avg_latency_ms": round(latency_sum / latency_count * 1000, 2) if latency_count > 0 else 0
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global request_count, latency_sum, latency_count
    start = time.time()
    request_count += 1

    try:
        contents = await file.read()
        result = predict_image(contents)
        latency = time.time() - start
        latency_sum += latency
        latency_count += 1

        logger.info(f"Prediction | class: {result['predicted_class']} | latency: {latency:.3f}s")
        return result

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
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