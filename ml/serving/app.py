"""FastAPI serving app.

Run:
    uvicorn ml.serving.app:app --workers 4 --loop uvloop --http httptools --port 8080

Endpoints:
    POST /v1/rank       — main ranking call, returns top-N slate w/ GSP prices
    GET  /healthz       — liveness
    GET  /readyz        — model loaded + (optional) Redis reachable
    GET  /metrics       — Prometheus exposition (latency histogram, cache stats)
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import ORJSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from ml.serving.auction_engine import AuctionEngine
from ml.serving.config import Config
from ml.serving.feature_store import FeatureStore
from ml.serving.predictor import Predictor
from ml.serving.schemas import RankRequest, RankResponse

REQUESTS = Counter("travelads_requests_total", "Rank requests", ["status"])
LATENCY = Histogram(
    "travelads_rank_latency_seconds",
    "End-to-end /v1/rank latency",
    buckets=(0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1.0),
)
CACHE_HITS = Gauge("travelads_cache_hits", "Feature cache hits (cumulative)", ["kind"])
CACHE_MISS = Gauge("travelads_cache_miss", "Feature cache misses (cumulative)", ["kind"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg: Config = app.state.config
    store = FeatureStore(
        cfg.redis_url, cfg.feature_ttl_s, cfg.user_cache_size, cfg.ad_cache_size
    )
    await store.start()
    predictor = Predictor(cfg.model_path)
    try:
        predictor.load()
    except Exception as e:  # don't crash before /readyz can report it
        app.state.startup_error = repr(e)
    engine = AuctionEngine(
        store, predictor, n_slots=cfg.n_slots, reserve_cpc=cfg.reserve_cpc
    )
    app.state.store = store
    app.state.predictor = predictor
    app.state.engine = engine
    try:
        yield
    finally:
        await store.close()


def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or Config.from_env()
    app = FastAPI(default_response_class=ORJSONResponse, lifespan=lifespan)
    app.state.config = cfg
    app.state.startup_error = None

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True}

    @app.get("/readyz")
    async def readyz() -> dict:
        if not app.state.predictor.ready:
            raise HTTPException(503, detail=app.state.startup_error or "model not loaded")
        return {"ok": True}

    @app.get("/metrics")
    async def metrics() -> Response:
        # Surface cache stats just-in-time so we don't need a background task.
        s = app.state.store.stats
        CACHE_HITS.labels(kind="user").set(s.user_hits)
        CACHE_HITS.labels(kind="ad").set(s.ad_hits)
        CACHE_MISS.labels(kind="user").set(s.user_miss)
        CACHE_MISS.labels(kind="ad").set(s.ad_miss)
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/rank", response_model=RankResponse)
    async def rank(req: RankRequest, request: Request) -> RankResponse:
        cfg: Config = request.app.state.config
        if len(req.candidates) > cfg.max_candidates:
            REQUESTS.labels(status="reject").inc()
            raise HTTPException(400, detail=f"too many candidates (>{cfg.max_candidates})")
        if not req.candidates:
            REQUESTS.labels(status="empty").inc()
            return RankResponse(request_id=req.request_id, results=[], latency_us=0)

        t0 = time.perf_counter_ns()
        try:
            results = await request.app.state.engine.rank(req.user, req.context, req.candidates)
        except Exception:
            REQUESTS.labels(status="error").inc()
            raise
        elapsed_ns = time.perf_counter_ns() - t0
        LATENCY.observe(elapsed_ns / 1e9)
        REQUESTS.labels(status="ok").inc()
        return RankResponse(
            request_id=req.request_id,
            results=results,
            latency_us=elapsed_ns // 1000,
        )

    return app


app = create_app()
