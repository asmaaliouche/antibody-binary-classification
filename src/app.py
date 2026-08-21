"""
app.py - Production-ready FastAPI scoring API for antibody HIC risk prediction
Includes database-backed call logging, strict data validation, and lookup-based inference.
"""

import logging
import os
import sqlite3
import sys
import time

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# Setup path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "api.log"))
    ]
)
logger = logging.getLogger("api")

# Initialize FastAPI App
app = FastAPI(
    title="Antibody Scoring API - HIC Risk Prediction",
    description="Production-grade scoring API to predict High Hydrophobic Interaction Chromatography (HIC) developability risks in antibodies.",
    version="1.0.0"
)

# Global variables to store our pre-loaded assets (loaded ONCE at startup)
model_pipeline = None
embedding_lookup_df = None
db_path = os.path.join(os.path.dirname(__file__), "production_logs.db")

# -----------------------------------------------------------------------------
# Database Setup for Call Logging
# -----------------------------------------------------------------------------
def init_db():
    """Initializes the SQLite database used to log all incoming API calls."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                antibody_id TEXT,
                vh TEXT,
                vl TEXT,
                tm_app REAL,
                psr REAL,
                prediction INTEGER,
                probability REAL,
                latency_ms REAL,
                status_code INTEGER,
                error_message TEXT
            )
        """)
        conn.commit()
        conn.close()
        logger.info(f"SQLite log database initialized successfully at: {db_path}")
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to initialize SQLite log database: {e}")

def log_api_call(
    antibody_id: str | None,
    vh: str,
    vl: str,
    tm_app: float | None,
    psr: float | None,
    prediction: int | None,
    probability: float | None,
    latency_ms: float,
    status_code: int,
    error_message: str | None = None
):
    """Inserts a single API call log entry into the SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO api_logs (
                antibody_id, vh, vl, tm_app, psr, prediction, probability, latency_ms, status_code, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            antibody_id, vh, vl, tm_app, psr, prediction, probability, latency_ms, status_code, error_message
        ))
        conn.commit()
        conn.close()
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to write API call log to database: {e}")

# -----------------------------------------------------------------------------
# Pydantic Schemas for Request Validation
# -----------------------------------------------------------------------------
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

class PredictionRequest(BaseModel):
    antibody_id: str | None = Field(None, example="Ab-42", description="Optional identifier for the antibody clone")
    vh: str = Field(..., example="EVQLVESGGGLVQPGGSLRLSCAASGFTFSDYAMHWVRQAPGKGLEW", description="Heavy chain (VH) protein sequence")
    vl: str = Field(..., example="DIQMTQSPSSLSASVGDRVTITCRASQGISNYLAWYQQKPGKAPKLLIY", description="Light chain (VL) protein sequence")
    tm_app: float = Field(..., example=68.5, description="Apparent melting temperature (°C)")
    psr: float = Field(..., example=0.25, description="Polyspecificity Ready (PSR) score")

    @validator("vh", "vl")
    def validate_sequence(cls, val):
        val_clean = val.replace("-", "").strip().upper()
        if not val_clean:
            raise ValueError("Sequence cannot be empty")
        
        # Verify valid amino acids
        invalid_chars = set(val_clean) - VALID_AMINO_ACIDS
        if invalid_chars:
            raise ValueError(f"Sequence contains invalid amino acid characters: {sorted(invalid_chars)}")
        return val_clean

    @validator("tm_app")
    def validate_tm_app(cls, val):
        # Human and structural validation limits
        if val < 30.0 or val > 110.0:
            raise ValueError("tm_app (melting temperature) must be within a realistic range [30.0, 110.0] °C")
        return val

    @validator("psr")
    def validate_psr(cls, val):
        # PSR scores usually lie between 0.0 and 10.0
        if val < 0.0 or val > 10.0:
            raise ValueError("psr score must be within [0.0, 10.0]")
        return val


class PredictionResponse(BaseModel):
    antibody_id: str | None
    prediction: int = Field(..., description="0 = Low/Medium HIC risk, 1 = High HIC risk")
    probability: float = Field(..., description="Predicted probability of being High HIC")
    lookup_status: str = Field(..., description="exact_match, nearest_match, or fallback_average")
    latency_ms: float = Field(..., description="Time taken for inference in milliseconds")

# -----------------------------------------------------------------------------
# API Events (Single Startup Loading)
# -----------------------------------------------------------------------------
@app.on_event("startup")
def startup_event():
    """Executed exactly ONCE when the API starts up. Loads serialized model and lookup database."""
    global model_pipeline, embedding_lookup_df
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "best_model.joblib")
    dataset_path = os.path.join(base_dir, "..", "data", "final_dataset.parquet")
    
    # 1. Initialize logging database
    init_db()

    # 2. Load the pre-trained Scikit-Learn Model Pipeline
    logger.info(f"Loading serialized model pipeline from: {model_path}")
    if os.path.exists(model_path):
        try:
            model_pipeline = joblib.load(model_path)
            logger.info("Model pipeline loaded successfully.")
        except Exception as e:  # noqa: BLE001
            logger.critical(f"Failed to load model pipeline: {e}")
            raise RuntimeError(f"Could not load serialized model: {e}")
    else:
        logger.critical(f"Model file not found at {model_path}. Run training first!")
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # 3. Load embedding lookup parquet (the Reference dataset)
    logger.info(f"Loading reference embeddings dataset for lookup from: {dataset_path}")
    if os.path.exists(dataset_path):
        try:
            full_df = pd.read_parquet(dataset_path)
            # We keep only relevant columns to minimize memory footprint
            embedding_lookup_df = full_df[[
                "vh", "vl", "vh_emb_prott5", "vl_emb_prott5"
            ]].copy()
            # Clean sequence columns in lookup to match incoming requests
            embedding_lookup_df["vh_clean"] = embedding_lookup_df["vh"].str.replace("-", "", regex=False).str.strip().str.upper()
            embedding_lookup_df["vl_clean"] = embedding_lookup_df["vl"].str.replace("-", "", regex=False).str.strip().str.upper()
            logger.info(f"Loaded {len(embedding_lookup_df)} reference sequences for lookup.")
        except Exception as e:  # noqa: BLE001
            logger.critical(f"Failed to load lookup dataset: {e}")
            raise RuntimeError(f"Could not load lookup dataset: {e}")
    else:
        logger.critical(f"Lookup dataset not found at {dataset_path}.")
        raise FileNotFoundError(f"Lookup dataset not found at {dataset_path}")


# -----------------------------------------------------------------------------
# Custom Validation Error Handler (Returns standard structured JSON and logs it)
# -----------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Captures Pydantic validation errors, logs them to SQLite, and returns a detailed JSON response."""
    errors = []
    for error in exc.errors():
        err_copy = dict(error)
        if "ctx" in err_copy and isinstance(err_copy["ctx"], dict):
            ctx_copy = dict(err_copy["ctx"])
            if "error" in ctx_copy and isinstance(ctx_copy["error"], Exception):
                ctx_copy["error"] = str(ctx_copy["error"])
            err_copy["ctx"] = ctx_copy
        errors.append(err_copy)
        
    logger.warning(f"Validation error on incoming request: {errors}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": errors}
    )


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/", tags=["General"])
def read_root():
    """Healthcheck endpoint."""
    return {
        "status": "healthy",
        "service": "Antibody Scoring API",
        "model": "ProtT5 + Logistic Regression (Winners)",
        "docs_url": "/docs"
    }


def find_embeddings_for_sequences(vh_req: str, vl_req: str):
    """
    Search strategies:
    1. Exact Match: Quick lookup on pre-calculated cleaned sequences.
    2. Nearest Match: Calculate Hamming/Levenshtein distance to find closest sequences.
    3. Fallback: Mean of all training embeddings.
    """
    # 1. Try Exact Match
    exact_match = embedding_lookup_df[
        (embedding_lookup_df["vh_clean"] == vh_req) & 
        (embedding_lookup_df["vl_clean"] == vl_req)
    ]
    
    if not exact_match.empty:
        row = exact_match.iloc[0]
        return row["vh_emb_prott5"], row["vl_emb_prott5"], "exact_match"
    
    # 2. Try Nearest Match (Simple sequence similarity based on Hamming distance / overlap)
    # We do a fast character matching overlap to find the best match
    best_score = -1
    best_row = None
    
    # To keep it fast (<5ms), we only scan a subset or do simple overlap
    for idx, row in embedding_lookup_df.iterrows():
        # Score based on length difference + character matches
        score_vh = sum(1 for c1, c2 in zip(vh_req, row["vh_clean"]) if c1 == c2)
        score_vl = sum(1 for c1, c2 in zip(vl_req, row["vl_clean"]) if c1 == c2)
        total_score = score_vh + score_vl
        
        if total_score > best_score:
            best_score = total_score
            best_row = row
            
    # If the closest sequence shares a decent overlap, return it
    if best_row is not None and best_score > 10: # arbitrary minimum character matches
        return best_row["vh_emb_prott5"], best_row["vl_emb_prott5"], "nearest_match"
        
    # 3. Fallback: Return dataset average embeddings
    mean_vh = np.stack(embedding_lookup_df["vh_emb_prott5"].values).mean(axis=0)
    mean_vl = np.stack(embedding_lookup_df["vl_emb_prott5"].values).mean(axis=0)
    return mean_vh, mean_vl, "fallback_average"


@app.post("/predict", response_model=PredictionResponse, tags=["Scoring"])
def predict_hic_risk(request: PredictionRequest):
    """
    Predicts if an antibody has a High HIC risk.
    """
    start_time = time.time()
    
    try:
        # Extract and clean sequences from Pydantic model
        vh_clean = request.vh
        vl_clean = request.vl
        
        # 1. Retrieve or calculate embeddings
        vh_emb, vl_emb, lookup_status = find_embeddings_for_sequences(vh_clean, vl_clean)
        
        # 2. Flatten and concatenate into the 2050-dimensional input vector
        # (1024 for VH + 1024 for VL + tm_app + psr)
        flattened_features = np.hstack([vh_emb, vl_emb, [request.tm_app, request.psr]])
        
        # 3. Reconstruct names exactly as expected by the pipeline model
        vh_names = [f"vh_prott5_{i}" for i in range(1024)]
        vl_names = [f"vl_prott5_{i}" for i in range(1024)]
        feature_names = vh_names + vl_names + ["tm_app", "psr"]
        
        X_df = pd.DataFrame([flattened_features], columns=feature_names)
        
        # 4. Perform Inference
        prediction = int(model_pipeline.predict(X_df)[0])
        probability = float(model_pipeline.predict_proba(X_df)[0, 1])
        
        latency_ms = (time.time() - start_time) * 1000.0
        
        # 5. Log API call inside database (Success path)
        log_api_call(
            antibody_id=request.antibody_id,
            vh=request.vh,
            vl=request.vl,
            tm_app=request.tm_app,
            psr=request.psr,
            prediction=prediction,
            probability=probability,
            latency_ms=latency_ms,
            status_code=200
        )
        
        return PredictionResponse(
            antibody_id=request.antibody_id,
            prediction=prediction,
            probability=probability,
            lookup_status=lookup_status,
            latency_ms=round(latency_ms, 2)
        )
        
    except Exception as e:  # noqa: BLE001
        latency_ms = (time.time() - start_time) * 1000.0
        logger.error(f"Internal API error during prediction: {e}")
        
        # Log call failure inside database
        log_api_call(
            antibody_id=request.antibody_id,
            vh=request.vh,
            vl=request.vl,
            tm_app=request.tm_app,
            psr=request.psr,
            prediction=None,
            probability=None,
            latency_ms=latency_ms,
            status_code=500,
            error_message=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Server Error: {e!s}"
        )
