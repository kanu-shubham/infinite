"""
predict.py
----------
A FastAPI REST API that serves hotel price predictions in real time.

HOW THE SERVING LAYER WORKS:
  1. Request comes in with hotel features (city, stars, amenities, …)
  2. API fetches LATEST feature values from the Feast online store
     (this is where "feature store → serving" becomes real)
  3. Preprocessor transforms raw values (same scaler fitted during training!)
  4. XGBoost model predicts price
  5. Response returned in < 50 ms

WHY FASTAPI?
  - Automatic request validation (Pydantic models)
  - Auto-generated /docs (Swagger UI) – great for demos
  - High performance (async, built on Starlette)
  - MLflow can deploy models as REST endpoints too (`mlflow models serve`),
    but building your own gives you full control over business logic.

RUN LOCALLY:
    uvicorn serving.predict:app --reload --port 8000
    → http://localhost:8000/docs
"""

import sys
from pathlib import Path

import joblib
import mlflow.xgboost
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parents[1]))

app = FastAPI(
    title="Hotel Price Prediction API",
    description="Predicts hotel price per night based on hotel attributes",
    version="1.0.0",
)

# ── Lazy-loaded globals (loaded once on first request) ────────────────────────
_model       = None
_preprocessor = None


def _load_artifacts():
    global _model, _preprocessor
    if _model is None:
        # Load from MLflow Model Registry (Production stage)
        _model = mlflow.xgboost.load_model("models:/hotel-price-predictor/Production")
    if _preprocessor is None:
        prep_path = Path(__file__).parents[1] / "models" / "artefacts" / "preprocessor.pkl"
        _preprocessor = joblib.load(prep_path)


# ── Request / Response schemas ────────────────────────────────────────────────
class HotelFeatures(BaseModel):
    hotel_id:        int   = Field(..., example=42)
    city:            str   = Field(..., example="Paris")
    category:        str   = Field(..., example="Luxury")
    star_rating:     int   = Field(..., ge=1, le=5, example=4)
    review_score:    float = Field(..., ge=1.0, le=10.0, example=8.5)
    num_reviews:     int   = Field(..., ge=0, example=1200)
    distance_km:     float = Field(..., ge=0.0, example=1.5)
    amenities:       int   = Field(..., ge=0, example=12)
    rooms_available: int   = Field(..., ge=0, example=5)


class PredictionResponse(BaseModel):
    hotel_id:         int
    predicted_price:  float = Field(..., description="Predicted price per night (USD)")
    model_version:    str


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Kubernetes/load-balancer liveness probe."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HotelFeatures):
    """
    Predict the price per night for a single hotel.

    The model was trained on numeric + categorical hotel attributes.
    Supply all fields; the API handles encoding internally.
    """
    try:
        _load_artifacts()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model not available: {exc}")

    import pandas as pd
    from training.preprocess import NUMERIC_FEATURES, CATEGORICAL_FEATURES

    row = pd.DataFrame([{
        "city":            features.city,
        "category":        features.category,
        "star_rating":     features.star_rating,
        "review_score":    features.review_score,
        "num_reviews":     features.num_reviews,
        "distance_km":     features.distance_km,
        "amenities":       features.amenities,
        "rooms_available": features.rooms_available,
    }])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    X = _preprocessor.transform(row)
    price = float(_model.predict(X)[0])

    return PredictionResponse(
        hotel_id=features.hotel_id,
        predicted_price=round(price, 2),
        model_version="Production",
    )


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(hotels: list[HotelFeatures]):
    """Predict prices for a list of hotels in a single call."""
    return [predict(h) for h in hotels]
