"""
Hotel Ranker — FastAPI serving layer.

Four prediction endpoints, each demonstrating a different serving strategy:

  POST /predict          Baseline sync inference (run_in_executor)
  POST /predict/cached   Same + LRU cache; cache hits take ~0.01 ms
  POST /predict/batch    Explicit client-side batching (N hotels → N scores)
  POST /predict/batched  Server-side micro-batching (looks single to client)

GET  /metrics            Rolling p50 / p95 / p99 + cache / batcher stats
GET  /health             Liveness probe

Throughput vs latency — the core tension
──────────────────────────────────────────
  /predict     lowest latency for a single request; throughput limited by
               how many threads can run inference in parallel.

  /predict/cached   lowest latency when hit; throughput ~ unlimited for
               cached inputs.  Miss path is identical to /predict.

  /predict/batch    highest raw throughput (amortises numpy overhead across
               N items); individual latency = 1 model call / N items.
               Client bears responsibility for assembling the batch.

  /predict/batched  trades up to max_wait_ms of added latency for
               throughput similar to /predict/batch — but the batching is
               transparent to the client (looks like a single request).
               Under high concurrency the batch fills fast, so the wait is
               negligible.  Under low concurrency most batches are size 1.

Workers and the GIL
────────────────────
With --workers 4, uvicorn forks 4 processes.  Each has its own GIL, model
copy, cache, and batcher.  True parallelism across cores at the cost of 4×
memory.  Alternative: 1 process + many async coroutines handles I/O-bound
work well, but for CPU-bound inference you need either:
  (a) multiple processes (--workers N)
  (b) a threadpool (run_in_executor, GIL released by numpy)
  (c) a separate inference server (Triton, TorchServe) over gRPC/HTTP
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from .batcher import MicroBatcher
from .cache   import PredictionCache
from .metrics  import LatencyTracker
from .model   import HotelRanker
from .schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
    Prediction,
)

# ── Singletons (one set per worker process) ────────────────────────────────────

ranker  = HotelRanker()
tracker = LatencyTracker(window_seconds=60)
cache   = PredictionCache(max_size=10_000, ttl_seconds=300)
batcher: MicroBatcher | None = None   # initialised in lifespan


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global batcher
    ranker.load()
    batcher = MicroBatcher(ranker, max_batch_size=32, max_wait_ms=10)
    await batcher.start()
    yield
    await batcher.stop()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Hotel Ranker",
    version="1.0.0",
    description=__doc__,
    lifespan=lifespan,
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": ranker.model_version}


# ── Strategy 1: single, async  ───────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> Any:
    """
    Single hotel ranking request.

    Inference runs in a thread (run_in_executor) so the event loop is never
    blocked.  At moderate concurrency this is your go-to baseline.

    Latency profile (single uvicorn worker, modern laptop):
      p50 ≈ 2–5 ms     p95 ≈ 5–15 ms     p99 ≈ 15–40 ms
    Numbers climb as the threadpool saturates under high concurrency.
    """
    t0          = time.perf_counter()
    hotel_dict  = req.hotel.model_dump()
    loop        = asyncio.get_running_loop()

    raw = await loop.run_in_executor(None, ranker.predict_single, hotel_dict)

    elapsed_ms = (time.perf_counter() - t0) * 1_000
    tracker.record(elapsed_ms)

    return PredictResponse(
        prediction=Prediction(
            click_probability=raw["click_probability"],
            rank_score=raw["rank_score"],
        ),
        model_version=ranker.model_version,
        latency_ms=round(elapsed_ms, 3),
    )


# ── Strategy 2: LRU cache  ────────────────────────────────────────────────────

@app.post("/predict/cached", response_model=PredictResponse)
async def predict_cached(req: PredictRequest) -> Any:
    """
    Prediction with LRU cache (TTL = 5 min, capacity = 10 000 entries).

    Cache key = BLAKE2s hash of the feature dict.  Identical feature vectors
    always map to the same key regardless of request_id.

    Hit path:  ~0.01 ms   (dict lookup, no inference)
    Miss path: same as /predict

    Observe with Locust: run a pool of N distinct hotels.  As N decreases
    (more repeated inputs), the hit rate climbs and p99 collapses.
    """
    t0         = time.perf_counter()
    hotel_dict = req.hotel.model_dump()

    hit    = cache.get(hotel_dict)
    if hit is not None:
        elapsed_ms = (time.perf_counter() - t0) * 1_000
        tracker.record(elapsed_ms)
        return PredictResponse(
            prediction=Prediction(**hit),
            model_version=ranker.model_version,
            latency_ms=round(elapsed_ms, 3),
            cache_hit=True,
        )

    loop = asyncio.get_running_loop()
    raw  = await loop.run_in_executor(None, ranker.predict_single, hotel_dict)
    cache.set(hotel_dict, {"click_probability": raw["click_probability"],
                           "rank_score":        raw["rank_score"]})

    elapsed_ms = (time.perf_counter() - t0) * 1_000
    tracker.record(elapsed_ms)
    return PredictResponse(
        prediction=Prediction(
            click_probability=raw["click_probability"],
            rank_score=raw["rank_score"],
        ),
        model_version=ranker.model_version,
        latency_ms=round(elapsed_ms, 3),
        cache_hit=False,
    )


# ── Strategy 3: explicit client-side batch ────────────────────────────────────

@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(req: BatchPredictRequest) -> Any:
    """
    Rank N hotels in a single model call.

    The client assembles the batch; the server runs one predict_proba() call.
    Best for: search results pages (rank 20 hotels at once) or offline scoring.

    Why one model call is cheaper than N individual calls:
      numpy/sklearn have fixed overhead per call (memory allocation, tree
      traversal setup).  A batch of 32 incurs that overhead once instead of
      32 times.  Throughput scales near-linearly with batch size up to the
      point where memory bandwidth becomes the bottleneck.

    Downside: the client must wait until it has all N hotel features before
    sending the request.  Not suitable for streaming / incremental results.
    """
    t0     = time.perf_counter()
    hotels = [h.model_dump() for h in req.hotels]
    loop   = asyncio.get_running_loop()

    raw = await loop.run_in_executor(None, ranker.predict_batch, hotels)

    elapsed_ms = (time.perf_counter() - t0) * 1_000
    tracker.record(elapsed_ms)

    return BatchPredictResponse(
        predictions=[Prediction(**p) for p in raw["predictions"]],
        batch_size=raw["batch_size"],
        model_version=ranker.model_version,
        latency_ms=round(elapsed_ms, 3),
    )


# ── Strategy 4: server-side micro-batching ────────────────────────────────────

@app.post("/predict/batched")
async def predict_batched(req: PredictRequest) -> Any:
    """
    Transparent server-side batching.

    The client sends a single hotel; the server groups concurrent requests
    into batches (up to 32, window = 10 ms) before calling the model.

    This is how TorchServe's "dynamic batching" and Triton's "preferred_batch_size"
    work.  The benefit is highest when:
      • Concurrency is high (batches fill quickly, window overhead is small)
      • Model has significant per-call overhead (GPT, vision models)

    Run the BatchedUser locust class to see avg_batch_size climb as you add
    concurrent users.  Watch how p99 stays bounded by max_wait_ms even when
    throughput reaches 2 000+ RPS.
    """
    if batcher is None:
        raise HTTPException(503, "Batcher not ready")

    t0         = time.perf_counter()
    hotel_dict = req.hotel.model_dump()

    pred = await batcher.predict(hotel_dict)

    elapsed_ms = (time.perf_counter() - t0) * 1_000
    tracker.record(elapsed_ms)

    return {
        "prediction":    pred,
        "model_version": ranker.model_version,
        "latency_ms":    round(elapsed_ms, 3),
        "avg_batch_size": batcher.avg_batch_size,
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

@app.get("/metrics")
def metrics() -> dict:
    """
    Rolling latency percentiles + cache / batcher diagnostics.

    Refresh this endpoint while running a Locust test to watch p99 climb
    as you increase the number of concurrent users.

    Key things to observe:
      1. p50 stays flat; p99 grows faster than p50 — tail latency is
         disproportionately affected by queuing (Little's Law).
      2. Cache hit_rate climbs as repeated inputs accumulate; p50 drops.
      3. batcher.avg_batch_size grows with concurrency, showing the
         efficiency benefit of dynamic batching.
    """
    return {
        "latency": tracker.stats(),
        "cache":   cache.stats,
        "batcher": {
            "total_batches":  batcher.total_batches if batcher else 0,
            "total_items":    batcher.total_items   if batcher else 0,
            "avg_batch_size": batcher.avg_batch_size if batcher else 0,
            "max_batch_size": batcher._max_batch     if batcher else 0,
            "max_wait_ms":    batcher._max_wait * 1_000 if batcher else 0,
        },
    }
