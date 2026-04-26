"""FastAPI service exposing the feed ranker."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from threading import RLock
from typing import Optional

from fastapi import FastAPI, HTTPException

from ml.feed_ranking import config as _config
from ml.feed_ranking.config import TrainConfig
from ml.feed_ranking.inference import Ranker
from ml.feed_ranking.schema import (
    Event,
    ModelInfo,
    Post,
    PredictRequest,
    PredictResponse,
    RankResponse,
    TrainRequest,
    TrainResponse,
    User,
)
from ml.feed_ranking.store import Store, get_store, set_store
from ml.feed_ranking.train import train as run_training

log = logging.getLogger("feed_ranking.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

_ranker: Optional[Ranker] = None
_ranker_lock = RLock()


def _get_ranker() -> Ranker:
    global _ranker
    with _ranker_lock:
        if _ranker is None:
            if not _config.DEFAULT_CHECKPOINT.exists():
                raise HTTPException(503, "no model loaded; POST /v1/model/train first")
            _ranker = Ranker.load(_config.DEFAULT_CHECKPOINT)
        return _ranker


def _set_ranker(r: Ranker) -> None:
    global _ranker
    with _ranker_lock:
        _ranker = r


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # Hydrate store from disk if a snapshot exists.
    if _config.DEFAULT_DATASET.exists():
        try:
            set_store(Store.load_json(_config.DEFAULT_DATASET))
            log.info("loaded store from %s", _config.DEFAULT_DATASET)
        except Exception:  # noqa: BLE001 — best-effort hydration
            log.exception("failed to load store from %s", _config.DEFAULT_DATASET)
    if _config.DEFAULT_CHECKPOINT.exists():
        try:
            _set_ranker(Ranker.load(_config.DEFAULT_CHECKPOINT))
            log.info("loaded model from %s", _config.DEFAULT_CHECKPOINT)
        except Exception:
            log.exception("failed to load model from %s", _config.DEFAULT_CHECKPOINT)
    yield


app = FastAPI(title="Feed Ranking", version="0.1.0", lifespan=_lifespan)


# -------- health / model info ----------------------------------------------

@app.get("/v1/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/v1/model/info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    if _ranker is None and not _config.DEFAULT_CHECKPOINT.exists():
        return ModelInfo(loaded=False)
    try:
        r = _get_ranker()
    except HTTPException:
        return ModelInfo(loaded=False)
    return ModelInfo(
        loaded=True,
        model_kind=r.metadata.get("model_kind"),
        feature_dim=r.metadata.get("feature_dim"),
        metrics=r.metadata.get("metrics"),
    )


# -------- entities ---------------------------------------------------------

@app.post("/v1/users", response_model=User)
def upsert_user(user: User) -> User:
    return get_store().upsert_user(user)


@app.post("/v1/posts", response_model=Post)
def upsert_post(post: Post) -> Post:
    store = get_store()
    if store.get_user(post.author_id) is None:
        raise HTTPException(400, f"unknown author_id: {post.author_id}")
    return store.upsert_post(post)


@app.post("/v1/connections")
def add_connection(payload: dict) -> dict:
    a, b = payload.get("a"), payload.get("b")
    if not a or not b:
        raise HTTPException(400, "payload must include 'a' and 'b'")
    store = get_store()
    if store.get_user(a) is None or store.get_user(b) is None:
        raise HTTPException(404, "both users must exist")
    store.connect(a, b)
    return {"a": a, "b": b, "connected": True}


# -------- events -----------------------------------------------------------

@app.post("/v1/events/impression", response_model=Event)
def log_impression(event: Event) -> Event:
    event = event.model_copy(update={"clicked": False})
    return _record_event(event)


@app.post("/v1/events/click", response_model=Event)
def log_click(event: Event) -> Event:
    event = event.model_copy(update={"clicked": True})
    return _record_event(event)


def _record_event(event: Event) -> Event:
    store = get_store()
    if store.get_user(event.user_id) is None:
        raise HTTPException(404, f"unknown user: {event.user_id}")
    if store.get_post(event.post_id) is None:
        raise HTTPException(404, f"unknown post: {event.post_id}")
    return store.add_event(event)


# -------- ranking ----------------------------------------------------------

@app.get("/v1/feed/{user_id}", response_model=RankResponse)
def feed(user_id: str, k: int = 20) -> RankResponse:
    store = get_store()
    if store.get_user(user_id) is None:
        raise HTTPException(404, f"unknown user: {user_id}")
    ranker = _get_ranker()
    ranked = ranker.rank(user_id, store=store, k=k)
    return RankResponse(user_id=user_id, ranked=ranked)


@app.post("/v1/model/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    ranker = _get_ranker()
    store = get_store()
    # Score per user (bucket by user_id) to share extractor work.
    by_user: dict[str, list[int]] = {}
    for i, item in enumerate(req.items):
        by_user.setdefault(item.user_id, []).append(i)
    scores = [0.0] * len(req.items)
    for uid, idxs in by_user.items():
        if store.get_user(uid) is None:
            raise HTTPException(404, f"unknown user: {uid}")
        post_ids = [req.items[i].post_id for i in idxs]
        for i, s in zip(idxs, ranker.score(uid, post_ids, store=store)):
            scores[i] = s
    return PredictResponse(scores=scores)


# -------- training ---------------------------------------------------------

@app.post("/v1/model/train", response_model=TrainResponse)
def train(req: TrainRequest) -> TrainResponse:
    cfg = TrainConfig()
    if req.epochs is not None:
        cfg.epochs = req.epochs
    if req.model_kind is not None:
        if req.model_kind not in {"mlp", "logreg"}:
            raise HTTPException(400, "model_kind must be 'mlp' or 'logreg'")
        cfg.model_kind = req.model_kind
    store = get_store()
    if not store.all_events():
        raise HTTPException(400, "store has no events; ingest events first")
    try:
        result = run_training(store, cfg, checkpoint_path=_config.DEFAULT_CHECKPOINT)
    except ValueError as e:
        raise HTTPException(400, str(e))
    # Reload ranker from the freshly written checkpoint.
    _set_ranker(Ranker.load(_config.DEFAULT_CHECKPOINT))
    # Best-effort persist store snapshot too.
    try:
        store.save_json(_config.DEFAULT_DATASET)
    except Exception:
        log.exception("failed to persist store snapshot")
    return result
