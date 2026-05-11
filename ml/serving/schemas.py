from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class UserPayload(BaseModel):
    user_id: str
    country: str = "XX"
    device: str = "desktop"
    segment: Optional[str] = None


class ContextPayload(BaseModel):
    destination: str
    lead_time_days: float = 0
    los: float = 1
    pax: float = 1
    hour: int = 12
    dow: int = 0


class CandidatePayload(BaseModel):
    ad_id: str
    advertiser_id: str
    bid_cpc: float = Field(gt=0)


class RankRequest(BaseModel):
    request_id: str
    user: UserPayload
    context: ContextPayload
    candidates: List[CandidatePayload]


class RankedAd(BaseModel):
    ad_id: str
    slot: int
    price_cpc: float
    pctr: float
    score: float


class RankResponse(BaseModel):
    request_id: str
    results: List[RankedAd]
    latency_us: int
