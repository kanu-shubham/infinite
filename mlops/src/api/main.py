"""
main.py
-------
Production FastAPI application that serves hotel price predictions.

Endpoints:
  GET  /           – service info
  GET  /health     – liveness + model status
  POST /predict    – single prediction
  GET  /metrics    – Prometheus scrape endpoint (auto-exposed)
  GET  /drift      – simple feature-drift report vs training baseline

Run locally (after `make train`):
    cd mlops/
    make serve
"""
import logging
import os
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.schemas import (
    DriftReport,
    HealthResponse,
    HotelFeatures,
    PredictionResponse,
    ConfidenceInterval,
)
from src.data_pipeline import FEATURE_COLUMNS, NUMERICAL_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

def _load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


CONFIG = _load_config()
MODEL_VERSION: str = CONFIG["model"]["version"]
MODEL_PATH: str = CONFIG["serving"]["model_path"]
DRIFT_THRESHOLD: float = CONFIG["monitoring"]["drift_threshold"]

# ── Prometheus custom metrics ─────────────────────────────────────────────────

PRICE_HISTOGRAM = Histogram(
    "hotel_price_prediction_usd",
    "Distribution of predicted hotel prices (USD)",
    buckets=[50, 100, 150, 200, 250, 300, 400, 500, 750, 1000],
)
PREDICTION_ERRORS = Counter(
    "hotel_price_prediction_errors_total",
    "Total prediction errors",
)

# ── App state (loaded once at startup) ───────────────────────────────────────

_model_pipeline = None
_baseline_stats: pd.DataFrame | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources once on startup; clean up on shutdown."""
    global _model_pipeline, _baseline_stats

    logger.info(f"Loading model from {MODEL_PATH} …")
    try:
        _model_pipeline = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully.")
    except FileNotFoundError:
        logger.error(
            f"Model not found at {MODEL_PATH}. "
            "Run `make train` before starting the API."
        )

    baseline_path = "data/baseline_stats.csv"
    if os.path.exists(baseline_path):
        _baseline_stats = pd.read_csv(baseline_path, index_col=0)
        logger.info("Baseline stats loaded for drift monitoring.")

    yield  # application runs here

    logger.info("Shutting down – releasing resources.")
    _model_pipeline = None


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Hotel Price Prediction API",
    description=(
        "Production ML API that predicts hotel price per night "
        "using an XGBoost model trained with scikit-learn Pipelines."
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Auto-instrument ALL routes with request count, latency histograms, etc.
Instrumentator().instrument(app).expose(app)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _features_to_df(features: HotelFeatures) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "city": features.city.value,
                "star_rating": features.star_rating,
                "amenities_count": features.amenities_count,
                "distance_to_center_km": features.distance_to_center_km,
                "review_score": features.review_score,
                "season": features.season.value,
                "rooms_available": features.rooms_available,
            }
        ]
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    return {
        "service": "Hotel Price Prediction API",
        "version": MODEL_VERSION,
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "drift": "/drift",
    }


@app.get("/health", response_model=HealthResponse, tags=["Ops"])
def health_check():
    return HealthResponse(
        status="healthy" if _model_pipeline is not None else "degraded",
        model_loaded=_model_pipeline is not None,
        model_version=MODEL_VERSION,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["ML"])
def predict(features: HotelFeatures):
    """
    Predict the price per night for a hotel given its attributes.

    Returns the predicted price and a ±10 % confidence interval
    (approximated from training RMSE).
    """
    if _model_pipeline is None:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=503, detail="Model not loaded – service degraded.")

    try:
        input_df = _features_to_df(features)
        raw_pred = float(_model_pipeline.predict(input_df)[0])
        price = max(round(raw_pred, 2), 50.0)

        # Track prediction distribution in Prometheus
        PRICE_HISTOGRAM.observe(price)

        margin = round(price * 0.10, 2)

        logger.info(
            f"Predicted ${price:.2f} for {features.city.value} "
            f"({features.season.value}, {features.star_rating}★)"
        )

        return PredictionResponse(
            predicted_price_per_night=price,
            model_version=MODEL_VERSION,
            confidence_interval=ConfidenceInterval(
                lower=round(price - margin, 2),
                upper=round(price + margin, 2),
            ),
        )

    except Exception as exc:
        PREDICTION_ERRORS.inc()
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/drift", response_model=list[DriftReport], tags=["Ops"])
def drift_report():
    """
    Compares live prediction-time feature distributions against the
    training baseline to surface data drift early.

    In production this would query a feature store or a rolling window
    of recent requests stored in a database.
    """
    if _baseline_stats is None:
        raise HTTPException(
            status_code=503,
            detail="Baseline stats not available. Run `make train` first.",
        )

    # Simulate 'current' data — in prod this comes from a request log table
    from src.data_pipeline import generate_hotel_data

    current_df = generate_hotel_data(n_samples=500, random_state=42)
    current_stats = current_df[NUMERICAL_FEATURES].describe()

    reports: list[DriftReport] = []
    for feature in NUMERICAL_FEATURES:
        baseline_mean = _baseline_stats.loc["mean", feature]
        current_mean = current_stats.loc["mean", feature]
        drift_pct = abs(current_mean - baseline_mean) / (abs(baseline_mean) + 1e-9)
        reports.append(
            DriftReport(
                feature=feature,
                baseline_mean=round(baseline_mean, 4),
                current_mean=round(current_mean, 4),
                drift_pct=round(drift_pct, 4),
                drifted=drift_pct > DRIFT_THRESHOLD,
            )
        )

    return reports
