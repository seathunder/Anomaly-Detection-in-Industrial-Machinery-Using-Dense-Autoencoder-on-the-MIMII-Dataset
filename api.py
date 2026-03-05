import os
import shutil
import tempfile
import logging
import yaml
import numpy as np
import tensorflow as tf
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Import necessary functions from your existing script
# We import specifically to avoid running the 'main' block of the other script
from mimii_baseline_rewrite import (
    load_config,
    load_scaler,
    predict_single_file_score,
    build_autoencoder,
    load_pickle
)

# --- Configuration & Globals ---
CONFIG_PATH = "baseline.yaml"
RESULT_DIR = "./result"
MODEL_DIR = "./model"
PICKLE_DIR = "./pickle"

# Global variables to hold the loaded model and settings
ml_context = {
    "model": None,
    "scaler": None,
    "feat_cfg": None,
    "run_cfg": None,
    "threshold": None
}

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - API - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_threshold():
    """Attempts to load the threshold from the results YAML file."""
    results_file = os.path.join(RESULT_DIR, "generalized_results.yaml")
    if os.path.exists(results_file):
        try:
            with open(results_file, "r") as f:
                data = yaml.safe_load(f)
                return data.get("threshold", float("nan"))
        except Exception as e:
            logger.error(f"Failed to read threshold: {e}")
    return float("nan")

def load_model_logic():
    """
    Replicates the robust model loading logic from your CLI script.
    Returns: (Loaded Model, Input Dimension)
    """
    model_full = os.path.join(MODEL_DIR, "generalized_model.keras")
    model_weights = os.path.join(MODEL_DIR, "generalized_model.weights.h5")
    train_pickle = os.path.join(PICKLE_DIR, "generalized_train_data.pickle")

    model = None

    # Strategy 1: Load full model
    if os.path.exists(model_full):
        logger.info(f"Loading full model from {model_full}")
        model = tf.keras.models.load_model(model_full)
        return model

    # Strategy 2: Load weights (requires knowing input_dim)
    if os.path.exists(model_weights):
        logger.info("Full model not found, attempting to load weights...")
        if os.path.exists(train_pickle):
            # We need to peek at the pickle to get input_dim
            try:
                # Note: This might be slow if the pickle is huge, but it's only once at startup
                train_data = load_pickle(train_pickle)
                input_dim = train_data.shape[1]
                logger.info(f"Inferred input_dim: {input_dim}")
                
                model = build_autoencoder(input_dim)
                model.load_weights(model_weights)
                return model
            except Exception as e:
                logger.error(f"Failed to load train pickle for dimensions: {e}")
        else:
            logger.error("Weights exist but train_data.pickle is missing. Cannot determine input dimension.")

    return None

# --- Lifespan Manager (Startup/Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Load Configuration
    cfg = load_config(CONFIG_PATH)
    ml_context["feat_cfg"] = cfg["feature"]
    ml_context["run_cfg"] = cfg.get("run", {})
    
    # 2. Load Scaler
    scaler_path = os.path.join(PICKLE_DIR, "generalized_scaler.pickle")
    ml_context["scaler"] = load_scaler(scaler_path)
    if ml_context["scaler"] is None:
        logger.warning("StandardScaler could not be loaded. Inference might be inaccurate.")

    # 3. Load Threshold
    ml_context["threshold"] = load_threshold()
    logger.info(f"Loaded Anomaly Threshold: {ml_context['threshold']}")

    # 4. Load Model
    ml_context["model"] = load_model_logic()
    
    if ml_context["model"] is None:
        logger.error("CRITICAL: Model failed to load. API will start but inference will fail.")
    else:
        logger.info("Model loaded successfully.")

    yield
    # Cleanup code (if any) goes here
    ml_context["model"] = None

# --- FastAPI App Definition ---
app = FastAPI(title="MIMII Anomaly Detection API", lifespan=lifespan)

# Allow Cross-Origin Resource Sharing (CORS) 
# This is crucial so your React Frontend (running on a different port) can talk to this Backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, allow all origins. Lock this down in prod.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Endpoints ---

@app.get("/")
async def root():
    """Health check endpoint."""
    status = "healthy" if ml_context["model"] is not None else "degraded (model missing)"
    return {
        "status": status,
        "threshold": ml_context["threshold"],
        "project": "MIMII Anomaly Detection"
    }

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Receives a .wav file, saves it temporarily, runs inference, and returns the result.
    """
    if ml_context["model"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    # Create a temporary file to save the uploaded audio
    # The existing 'predict_single_file_score' function requires a file path on disk.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        try:
            # Write uploaded bytes to temp file
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        finally:
            file.file.close()

    try:
        # Run Inference
        logger.info(f"Processing file: {file.filename}")
        score = predict_single_file_score(
            model=ml_context["model"],
            filepath=temp_path,
            feat_cfg=ml_context["feat_cfg"],
            scaler=ml_context["scaler"],
            batch_size=ml_context["run_cfg"].get("batch_predict_size", 256)
        )

        # Logic for Verdict
        threshold = ml_context["threshold"]
        verdict = "UNKNOWN"
        
        if np.isnan(threshold):
            verdict = "THRESHOLD_MISSING"
        else:
            verdict = "ABNORMAL" if score > threshold else "NORMAL"

        return {
            "filename": file.filename,
            "anomaly_score": float(score),
            "threshold": float(threshold),
            "verdict": verdict,
            "is_abnormal": bool(score > threshold) if not np.isnan(threshold) else False
        }

    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    
    finally:
        # Clean up the temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)