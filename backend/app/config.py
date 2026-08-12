"""Application settings, resolved once at import time."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Where trained models and run metadata are persisted between restarts.
ARTIFACTS_DIR = Path(os.getenv("ML_ARTIFACTS_DIR", BASE_DIR / "artifacts"))
RUNS_DIR = ARTIFACTS_DIR / "runs"

# Size of the synthetic hotel-bookings dataset the pipeline trains on.
DATASET_ROWS = int(os.getenv("ML_DATASET_ROWS", "6000"))
DATASET_SEED = int(os.getenv("ML_DATASET_SEED", "42"))

# Browser origins allowed to call the API (CRA dev server by default).
CORS_ORIGINS = os.getenv(
    "ML_CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
).split(",")

# Cap on concurrent/stored runs so the artifacts dir cannot grow unbounded.
MAX_STORED_RUNS = int(os.getenv("ML_MAX_STORED_RUNS", "50"))

# ── Online feature store ─────────────────────────────────────────────────────
# Empty URL falls back to an in-process dict, so the app still boots without
# Redis. That fallback is for tests and local work only: it is per-process, so
# with more than one API replica each would hold a different view.
REDIS_URL = os.getenv("ML_REDIS_URL", "redis://localhost:6379/0")

# How long a materialised feature row stays servable. This is the *hard*
# expiry; `staleness_seconds` on each response is the observable age, and the
# two answer different questions — "is it gone" versus "how old is it".
FEATURE_TTL_SECONDS = int(os.getenv("ML_FEATURE_TTL_SECONDS", "3600"))

# Refuse to serve a feature row older than this even if the TTL has not fired.
# Set to 0 to disable the check.
FEATURE_MAX_STALENESS_SECONDS = int(os.getenv("ML_FEATURE_MAX_STALENESS", "0"))

# ── Model cache ──────────────────────────────────────────────────────────────
# Number of fitted pipelines held in memory. Loading from disk on every request
# costs tens of milliseconds and dominates inference latency.
MODEL_CACHE_SIZE = int(os.getenv("ML_MODEL_CACHE_SIZE", "4"))
