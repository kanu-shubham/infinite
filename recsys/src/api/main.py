"""
main.py
-------
FastAPI serving layer for the two-tower recommendation system.

Endpoints:
  GET  /           – service info
  GET  /health     – liveness + models status
  POST /recommend  – get top-K hotel recommendations for a user
  GET  /metrics    – Prometheus scrape endpoint

Two-stage inference per request:
  1. Encode user features → user embedding (Two-Tower user tower)
  2. Retrieve top-N candidates via embedding index dot-product search
  3. Re-rank candidates with XGBoost
  4. Return top-K results

Run locally (after `make train`):
    cd recsys/
    make serve
"""
import logging
import os
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.schemas import (
    HealthResponse,
    HotelRecommendation,
    RecommendationResponse,
    UserContext,
)
from src.model.towers import TwoTowerModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


CONFIG = _load_config()
MODEL_VERSION: str = CONFIG["model"]["version"]
TOP_K: int = CONFIG["serving"]["top_k"]
N_CANDIDATES: int = CONFIG["ranker"]["n_candidates"]

# ── Prometheus metrics ────────────────────────────────────────────────────────

RETRIEVAL_LATENCY = Histogram(
    "recsys_retrieval_latency_seconds",
    "Time spent in Stage 1 (Two-Tower retrieval)",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
)
RANKING_LATENCY = Histogram(
    "recsys_ranking_latency_seconds",
    "Time spent in Stage 2 (XGBoost ranking)",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
)
RECOMMENDATION_ERRORS = Counter(
    "recsys_recommendation_errors_total",
    "Total recommendation errors",
)

# ── Model state ───────────────────────────────────────────────────────────────

_tower_model = None
_user_encoder = None
_hotel_encoder = None
_embedding_index = None
_ranker = None
_hotel_df = None


def _models_ready() -> bool:
    return all([
        _tower_model, _user_encoder, _hotel_encoder,
        _embedding_index, _ranker, _hotel_df is not None,
    ])


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _tower_model, _user_encoder, _hotel_encoder
    global _embedding_index, _ranker, _hotel_df

    model_dir = CONFIG["serving"]["model_path"]
    logger.info(f"Loading models from {model_dir} …")

    try:
        _user_encoder = joblib.load(os.path.join(model_dir, "user_encoder.pkl"))
        _hotel_encoder = joblib.load(os.path.join(model_dir, "hotel_encoder.pkl"))
        _embedding_index = joblib.load(os.path.join(model_dir, "embedding_index.pkl"))
        _ranker = joblib.load(os.path.join(model_dir, "ranker.pkl"))
        tower_cfg = joblib.load(os.path.join(model_dir, "tower_config.pkl"))

        _tower_model = TwoTowerModel(**tower_cfg)
        _tower_model.load_state_dict(
            torch.load(os.path.join(model_dir, "two_tower.pt"), map_location="cpu")
        )
        _tower_model.eval()

        # Load hotel feature table so ranker can build features per candidate
        from src.data_generator import generate_dataset
        _, hotels, _ = generate_dataset()
        _hotel_df = hotels

        logger.info("All models loaded successfully.")
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}. Run `make train` first.")

    yield

    logger.info("Shutting down.")
    _tower_model = None


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Hotel Recommendation API",
    description=(
        "Two-stage recommendation system: "
        "Two-Tower retrieval → XGBoost re-ranking"
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

Instrumentator().instrument(app).expose(app)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    return {
        "service": "Hotel Recommendation API",
        "version": MODEL_VERSION,
        "architecture": "Two-Tower retrieval + XGBoost re-ranking",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


@app.get("/health", response_model=HealthResponse, tags=["Ops"])
def health():
    ready = _models_ready()
    return HealthResponse(
        status="healthy" if ready else "degraded",
        models_loaded=ready,
        model_version=MODEL_VERSION,
    )


@app.post("/recommend", response_model=RecommendationResponse, tags=["ML"])
def recommend(user: UserContext):
    """
    Return top-K hotel recommendations for a user.

    Stage 1 (retrieval): encode user → embedding → dot-product against
    all hotel embeddings → top-N candidates.

    Stage 2 (ranking): for each candidate, build rich features
    (user features + hotel features + tower similarity score) and score
    with XGBoost → return top-K sorted by score.
    """
    if not _models_ready():
        RECOMMENDATION_ERRORS.inc()
        raise HTTPException(status_code=503, detail="Models not loaded.")

    try:
        import time

        # Build user feature row as a single-row DataFrame
        user_dict = {
            "age_group": user.age_group.value,
            "travel_type": user.travel_type.value,
            "price_sensitivity": user.price_sensitivity,
            "review_weight": user.review_weight,
            "prefers_city_center": user.prefers_city_center,
            "loyalty_tier": user.loyalty_tier.value,
        }
        user_row = pd.DataFrame([user_dict])
        user_enc = _user_encoder.transform(user_row)

        # ── Stage 1: retrieval ────────────────────────────────────────────────
        t0 = time.perf_counter()
        with torch.no_grad():
            user_emb = (
                _tower_model.get_user_embedding(
                    torch.tensor(user_enc, dtype=torch.float32)
                )
                .numpy()
                .squeeze()
            )
        candidate_ids = _embedding_index.get_top_k(user_emb, k=N_CANDIDATES)
        retrieval_time = time.perf_counter() - t0
        RETRIEVAL_LATENCY.observe(retrieval_time)

        # ── Stage 2: ranking ─────────────────────────────────────────────────
        t1 = time.perf_counter()
        ranked_hotel_ids = _ranker.rank_candidates(
            candidate_hotel_ids=candidate_ids,
            user_encoded=user_enc,
            hotel_df=_hotel_df,
            hotel_encoder=_hotel_encoder,
            tower_model=_tower_model,
            top_k=TOP_K,
        )
        ranking_time = time.perf_counter() - t1
        RANKING_LATENCY.observe(ranking_time)

        logger.info(
            f"Recommended {len(ranked_hotel_ids)} hotels "
            f"(retrieval: {retrieval_time*1000:.1f}ms, "
            f"ranking: {ranking_time*1000:.1f}ms)"
        )

        recommendations = [
            HotelRecommendation(
                hotel_id=int(hid),
                rank=rank,
                relevance_score=round(float(TOP_K - rank) / TOP_K, 3),
            )
            for rank, hid in enumerate(ranked_hotel_ids, 1)
        ]

        return RecommendationResponse(
            recommendations=recommendations,
            n_candidates_retrieved=len(candidate_ids),
            n_final_recommendations=len(recommendations),
            model_version=MODEL_VERSION,
        )

    except Exception as exc:
        RECOMMENDATION_ERRORS.inc()
        logger.exception("Recommendation failed.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
