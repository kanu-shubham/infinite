"""
schemas.py
----------
Pydantic v2 request / response models for the prediction API.

Strict validation here means bad inputs are rejected at the HTTP boundary
before they ever reach the model — a critical production safety practice.
"""
from enum import Enum

from pydantic import BaseModel, Field


class City(str, Enum):
    new_york = "New York"
    london = "London"
    paris = "Paris"
    tokyo = "Tokyo"
    dubai = "Dubai"


class Season(str, Enum):
    peak = "peak"
    off_peak = "off-peak"
    shoulder = "shoulder"


class HotelFeatures(BaseModel):
    city: City
    star_rating: int = Field(..., ge=3, le=5, description="Hotel star rating (3–5)")
    amenities_count: int = Field(..., ge=1, le=50, description="Number of amenities offered")
    distance_to_center_km: float = Field(
        ..., ge=0.1, le=30.0, description="Distance to city centre in km"
    )
    review_score: float = Field(..., ge=1.0, le=10.0, description="Guest review score (1–10)")
    season: Season
    rooms_available: int = Field(..., ge=1, le=100, description="Rooms currently available")

    model_config = {
        "json_schema_extra": {
            "example": {
                "city": "London",
                "star_rating": 4,
                "amenities_count": 15,
                "distance_to_center_km": 2.5,
                "review_score": 8.7,
                "season": "peak",
                "rooms_available": 12,
            }
        }
    }


class ConfidenceInterval(BaseModel):
    lower: float
    upper: float


class PredictionResponse(BaseModel):
    predicted_price_per_night: float
    currency: str = "USD"
    model_version: str
    confidence_interval: ConfidenceInterval


class HealthResponse(BaseModel):
    status: str          # "healthy" | "degraded"
    model_loaded: bool
    model_version: str


class DriftReport(BaseModel):
    feature: str
    baseline_mean: float
    current_mean: float
    drift_pct: float
    drifted: bool
