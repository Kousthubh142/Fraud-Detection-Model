"""
api/main.py

FastAPI fraud detection API.
Serves the Production XGBoost model with SHAP explainability.

Run locally:
    uvicorn api.main:app --reload --port 8000

API docs auto-generated at:
    http://localhost:8000/docs      ← Swagger UI
    http://localhost:8000/redoc     ← ReDoc
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.predictor import predictor
from api.schemas import (
    BatchPredictionResponse,
    BatchTransactionRequest,
    FeatureContribution,
    HealthResponse,
    PredictionResponse,
    TransactionRequest,
)


# ── Lifespan — load model once at startup ────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the Production model when the server starts."""
    print("Starting up — loading fraud detection model...")
    predictor.load()
    print("API ready!")
    yield
    print("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Fraud Detection API",
    description="""
Real-time credit card fraud detection using XGBoost + SHAP explainability.

## Features
- Scores individual transactions in < 50ms
- Returns fraud probability + risk level (LOW / MEDIUM / HIGH)
- Explains every prediction with top 5 SHAP feature contributions
- Batch endpoint for scoring up to 100 transactions at once
- Model version tracking — always serves the Production model
    """,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Check API health and model version."""
    return HealthResponse(
        status        = "ok",
        model_name    = "fraud_detector",
        model_version = predictor.model_version,
        model_stage   = predictor.model_stage,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(transaction: TransactionRequest):
    """
    Score a single transaction for fraud.

    Returns fraud probability (0–1), risk level, and top 5 SHAP
    feature contributions explaining why the model flagged it.
    """
    try:
        raw    = transaction.model_dump()
        result = predictor.predict(raw)
        return PredictionResponse(
            fraud_probability = result["fraud_probability"],
            is_fraud          = result["is_fraud"],
            risk_level        = result["risk_level"],
            threshold_used    = result["threshold_used"],
            top_features      = [FeatureContribution(**f) for f in result["top_features"]],
            latency_ms        = result["latency_ms"],
            model_version     = result["model_version"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchTransactionRequest):
    """
    Score up to 100 transactions in a single request.
    Returns predictions for all transactions plus aggregate stats.
    """
    start = time.time()
    try:
        raw_txns     = [t.model_dump() for t in request.transactions]
        results      = predictor.predict_batch(raw_txns)
        predictions  = [
            PredictionResponse(
                fraud_probability = r["fraud_probability"],
                is_fraud          = r["is_fraud"],
                risk_level        = r["risk_level"],
                threshold_used    = r["threshold_used"],
                top_features      = [FeatureContribution(**f) for f in r["top_features"]],
                latency_ms        = r["latency_ms"],
                model_version     = r["model_version"],
            )
            for r in results
        ]
        return BatchPredictionResponse(
            predictions  = predictions,
            total        = len(predictions),
            fraud_count  = sum(1 for p in predictions if p.is_fraud),
            latency_ms   = round((time.time() - start) * 1000, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info", tags=["System"])
async def model_info():
    """Return metadata about the currently loaded model."""
    return {
        "model_name":    "fraud_detector",
        "model_version": predictor.model_version,
        "model_stage":   predictor.model_stage,
        "threshold":     predictor.threshold,
        "features":      36,
        "framework":     "XGBoost",
        "explainability": "SHAP TreeExplainer",
    }


@app.get("/", tags=["System"])
async def root():
    return {
        "message": "Fraud Detection API",
        "docs":    "http://localhost:8000/docs",
        "health":  "http://localhost:8000/health",
    }