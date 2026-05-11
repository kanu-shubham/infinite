"""
Pydantic request / response contracts.

Keeping schemas in a dedicated module makes it easy to version the API
(v1 vs v2 schemas) and to generate OpenAPI docs without touching business logic.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class HotelFeatures(BaseModel):
    price:          float = Field(..., gt=0,   description="USD per night")
    rating:         float = Field(..., ge=1.0, le=5.0)
    review_count:   int   = Field(..., ge=0)
    amenity_count:  int   = Field(..., ge=0)
    location_score: float = Field(..., ge=0.0, le=1.0,
                                  description="0 = remote, 1 = city-centre")


class PredictRequest(BaseModel):
    hotel:      HotelFeatures
    request_id: Optional[str] = None


class BatchPredictRequest(BaseModel):
    hotels:     List[HotelFeatures] = Field(..., min_length=1, max_length=256)
    request_id: Optional[str] = None


class Prediction(BaseModel):
    click_probability: float
    rank_score:        float  # same as probability; named for the ranking use-case


class PredictResponse(BaseModel):
    prediction:    Prediction
    model_version: str
    latency_ms:    float
    cache_hit:     bool = False


class BatchPredictResponse(BaseModel):
    predictions:   List[Prediction]
    batch_size:    int
    model_version: str
    latency_ms:    float
