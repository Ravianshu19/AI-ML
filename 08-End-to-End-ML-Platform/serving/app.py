"""FastAPI model serving application.

Serves the production churn prediction model from the MLflow
model registry. Supports single and batch predictions.

Usage:
    uvicorn serving.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ─── Request / Response Schemas ──────────────────────────────────────
class CustomerFeatures(BaseModel):
    """Input features for a single customer prediction."""
    age: int = Field(..., ge=18, le=100, description="Customer age")
    tenure_months: int = Field(..., ge=0, description="Months as customer")
    senior_citizen: int = Field(..., ge=0, le=1)
    monthly_logins: int = Field(..., ge=0)
    monthly_transactions: int = Field(..., ge=0)
    support_tickets_30d: int = Field(..., ge=0)
    avg_session_duration_min: float = Field(..., ge=0)
    days_since_last_login: int = Field(..., ge=0)
    product_usage_score: float = Field(..., ge=0, le=100)
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    late_payments_6m: int = Field(..., ge=0)
    has_autopay: int = Field(..., ge=0, le=1)

    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "tenure_months": 24,
                "senior_citizen": 0,
                "monthly_logins": 15,
                "monthly_transactions": 8,
                "support_tickets_30d": 1,
                "avg_session_duration_min": 12.5,
                "days_since_last_login": 3,
                "product_usage_score": 72.5,
                "monthly_charges": 65.0,
                "total_charges": 1560.0,
                "late_payments_6m": 0,
                "has_autopay": 1,
            }
        }


class PredictionResponse(BaseModel):
    """Prediction output."""
    customer_id: str | None = None
    churn_prediction: int
    churn_probability: float
    risk_level: str  # Low, Medium, High
    model_version: str
    timestamp: str


class BatchPredictionRequest(BaseModel):
    """Batch prediction input."""
    customers: list[CustomerFeatures]


class BatchPredictionResponse(BaseModel):
    """Batch prediction output."""
    predictions: list[PredictionResponse]
    total: int
    high_risk_count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_name: str
    model_version: str
    timestamp: str


# ─── Global Model State ──────────────────────────────────────────────
model_state: dict[str, Any] = {
    "model": None,
    "model_name": "",
    "model_version": "",
}


def load_config() -> dict:
    config_path = "config.yaml"
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def load_model() -> None:
    """Load the production model from MLflow registry."""
    config = load_config()
    tracking_uri = config.get("mlflow", {}).get(
        "tracking_uri",
        os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    )
    model_name = config.get("mlflow", {}).get(
        "model_name",
        os.getenv("MODEL_NAME", "churn_classifier"),
    )
    model_stage = config.get("serving", {}).get(
        "model_stage",
        os.getenv("MODEL_STAGE", "Production"),
    )

    mlflow.set_tracking_uri(tracking_uri)

    try:
        model_uri = f"models:/{model_name}/{model_stage}"
        model_state["model"] = mlflow.sklearn.load_model(model_uri)
        model_state["model_name"] = model_name
        model_state["model_version"] = model_stage
        logger.info("Model loaded: %s (%s)", model_name, model_stage)
    except Exception as e:
        logger.warning("Could not load model from registry: %s", e)
        logger.info("Server will start without a model. Use /health to check.")


# ─── App Lifecycle ───────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    load_model()
    yield


app = FastAPI(
    title="🔮 Churn Prediction API",
    description="Real-time customer churn predictions served from MLflow model registry.",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Endpoints ───────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API and model health."""
    return HealthResponse(
        status="healthy" if model_state["model"] is not None else "degraded",
        model_loaded=model_state["model"] is not None,
        model_name=model_state["model_name"],
        model_version=model_state["model_version"],
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: CustomerFeatures):
    """Predict churn for a single customer."""
    if model_state["model"] is None:
        raise HTTPException(503, "Model not loaded. Check /health.")

    df = pd.DataFrame([features.model_dump()])
    prediction = model_state["model"].predict(df)[0]
    probability = (
        model_state["model"].predict_proba(df)[0][1]
        if hasattr(model_state["model"], "predict_proba")
        else float(prediction)
    )

    risk_level = (
        "High" if probability > 0.7
        else "Medium" if probability > 0.4
        else "Low"
    )

    return PredictionResponse(
        churn_prediction=int(prediction),
        churn_probability=round(float(probability), 4),
        risk_level=risk_level,
        model_version=model_state["model_version"],
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict churn for a batch of customers."""
    if model_state["model"] is None:
        raise HTTPException(503, "Model not loaded. Check /health.")

    df = pd.DataFrame([c.model_dump() for c in request.customers])
    predictions = model_state["model"].predict(df)
    probabilities = (
        model_state["model"].predict_proba(df)[:, 1]
        if hasattr(model_state["model"], "predict_proba")
        else predictions.astype(float)
    )

    results = []
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        risk = "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
        results.append(
            PredictionResponse(
                customer_id=f"batch_{i}",
                churn_prediction=int(pred),
                churn_probability=round(float(prob), 4),
                risk_level=risk,
                model_version=model_state["model_version"],
                timestamp=datetime.utcnow().isoformat(),
            )
        )

    high_risk = sum(1 for r in results if r.risk_level == "High")

    return BatchPredictionResponse(
        predictions=results,
        total=len(results),
        high_risk_count=high_risk,
    )


@app.post("/reload")
async def reload_model():
    """Reload the model from the registry (e.g., after promotion)."""
    load_model()
    return {
        "status": "reloaded",
        "model_loaded": model_state["model"] is not None,
        "model_name": model_state["model_name"],
    }
