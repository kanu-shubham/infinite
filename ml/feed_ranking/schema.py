"""Pydantic models for the API and store."""
from __future__ import annotations

import time
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from .config import POST_TYPES, TOPICS


def _now() -> float:
    return time.time()


class User(BaseModel):
    user_id: str
    created_at: float = Field(default_factory=_now)
    num_connections: int = 0
    num_followers: int = 0
    topic_affinities: dict[str, float] = Field(default_factory=dict)

    @field_validator("topic_affinities")
    @classmethod
    def _validate_topics(cls, v: dict[str, float]) -> dict[str, float]:
        bad = set(v) - set(TOPICS)
        if bad:
            raise ValueError(f"unknown topics: {sorted(bad)}")
        return v


class Post(BaseModel):
    post_id: str
    author_id: str
    created_at: float = Field(default_factory=_now)
    post_type: str
    topic: str
    text_length: int = 0
    num_likes: int = 0
    num_comments: int = 0
    num_shares: int = 0

    @field_validator("post_type")
    @classmethod
    def _vt(cls, v: str) -> str:
        if v not in POST_TYPES:
            raise ValueError(f"post_type must be one of {POST_TYPES}")
        return v

    @field_validator("topic")
    @classmethod
    def _vc(cls, v: str) -> str:
        if v not in TOPICS:
            raise ValueError(f"topic must be one of {TOPICS}")
        return v


class Event(BaseModel):
    """A single (user, post, click?) impression event."""

    user_id: str
    post_id: str
    timestamp: float = Field(default_factory=_now)
    clicked: bool = False
    # Cached author connection at event time. Optional — derived if absent.
    is_connected: Optional[bool] = None


class RankRequest(BaseModel):
    user_id: str
    candidate_post_ids: Optional[list[str]] = None  # None => all known posts
    k: int = 20


class ScoredPost(BaseModel):
    post_id: str
    score: float


class RankResponse(BaseModel):
    user_id: str
    ranked: list[ScoredPost]


class PredictItem(BaseModel):
    user_id: str
    post_id: str


class PredictRequest(BaseModel):
    items: list[PredictItem]


class PredictResponse(BaseModel):
    scores: list[float]


class TrainRequest(BaseModel):
    epochs: Optional[int] = None
    model_kind: Optional[str] = None  # "mlp" | "logreg"


class TrainResponse(BaseModel):
    model_kind: str
    epochs: int
    train_loss: float
    val_auc: float
    val_nce: float
    test_auc: float
    test_nce: float
    replay_topk_match_rate: float
    n_train: int
    n_val: int
    n_test: int


class ModelInfo(BaseModel):
    loaded: bool
    model_kind: Optional[str] = None
    feature_dim: Optional[int] = None
    metrics: Optional[dict] = None
