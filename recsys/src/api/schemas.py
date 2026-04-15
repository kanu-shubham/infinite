"""
schemas.py — Pydantic request/response models for the recommendation API.
"""
from pydantic import BaseModel, Field
from enum import Enum


class TravelType(str, Enum):
    business = "business"
    leisure = "leisure"
    family = "family"
    solo = "solo"


class AgeGroup(str, Enum):
    g18_25 = "18-25"
    g26_35 = "26-35"
    g36_50 = "36-50"
    g51_plus = "51+"


class LoyaltyTier(str, Enum):
    bronze = "bronze"
    silver = "silver"
    gold = "gold"


class UserContext(BaseModel):
    """What we know about the user at request time."""
    age_group: AgeGroup
    travel_type: TravelType
    price_sensitivity: float = Field(..., ge=0.0, le=1.0)
    review_weight: float = Field(..., ge=0.0, le=1.0)
    prefers_city_center: int = Field(..., ge=0, le=1)
    loyalty_tier: LoyaltyTier

    model_config = {
        "json_schema_extra": {
            "example": {
                "age_group": "26-35",
                "travel_type": "business",
                "price_sensitivity": 0.3,
                "review_weight": 0.8,
                "prefers_city_center": 1,
                "loyalty_tier": "gold",
            }
        }
    }


class HotelRecommendation(BaseModel):
    hotel_id: int
    rank: int                  # 1 = top recommendation
    relevance_score: float     # ranker score (higher = more relevant)


class RecommendationResponse(BaseModel):
    recommendations: list[HotelRecommendation]
    n_candidates_retrieved: int   # how many Stage 1 returned
    n_final_recommendations: int  # how many Stage 2 kept
    model_version: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    model_version: str
