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
