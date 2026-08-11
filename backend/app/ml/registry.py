"""Run store and model registry.

Runs live in memory for fast reads and are mirrored to disk as JSON so a
restart does not lose the history; fitted pipelines are pickled next to
their metadata. Training executes on a worker thread, so every mutation
goes through a lock.
"""

import json
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import joblib

from ..config import MAX_STORED_RUNS, RUNS_DIR

MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "run.json"

# Fields that make a run summary; the detail payload (curves, scatter, logs)
# is deliberately left out of list responses.
SUMMARY_FIELDS = (
    "run_id",
    "name",
    "status",
    "created_at",
    "finished_at",
    "duration_ms",
    "config",
    "headline_metric",
    "error",
)

_lock = threading.RLock()
_runs: Dict[str, Dict[str, Any]] = {}
_loaded = False


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id


def _persist(run: Dict[str, Any]) -> None:
    directory = _run_dir(run["run_id"])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / METADATA_FILENAME
    # Write-then-rename so a crash mid-write cannot leave truncated JSON.
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(run, indent=2, default=str))
    temporary.replace(path)


def load_from_disk() -> None:
    """Rehydrate the in-memory store. Safe to call more than once."""
    global _loaded
    with _lock:
        if _loaded:
            return
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        for directory in sorted(RUNS_DIR.iterdir()):
            metadata = directory / METADATA_FILENAME
            if not directory.is_dir() or not metadata.exists():
                continue
            try:
                run = json.loads(metadata.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            # A run that was mid-flight when the process died can never finish.
            if run.get("status") in {"queued", "running"}:
                run["status"] = "failed"
                run["error"] = "Interrupted by a server restart."
                _persist(run)
            _runs[run["run_id"]] = run
        _loaded = True


def next_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    with _lock:
        suffix = 1
        run_id = f"run-{stamp}"
        while run_id in _runs:
            suffix += 1
            run_id = f"run-{stamp}-{suffix}"
        return run_id


def create(run: Dict[str, Any]) -> Dict[str, Any]:
    with _lock:
        _runs[run["run_id"]] = run
        _persist(run)
        _evict_locked()
        return dict(run)


def update(run_id: str, mutate: Callable[[Dict[str, Any]], None]) -> Optional[Dict[str, Any]]:
    """Apply `mutate` to the stored run under the lock, then persist it."""
    with _lock:
        run = _runs.get(run_id)
        if run is None:
            return None
        mutate(run)
        _persist(run)
        return dict(run)


def get(run_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        run = _runs.get(run_id)
        return dict(run) if run else None


def summary(run: Dict[str, Any]) -> Dict[str, Any]:
    return {field: run.get(field) for field in SUMMARY_FIELDS}


def list_summaries(status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    with _lock:
        runs = sorted(_runs.values(), key=lambda r: r["created_at"], reverse=True)
    if status:
        runs = [r for r in runs if r["status"] == status]
    return [summary(run) for run in runs[:limit]]


def delete(run_id: str) -> bool:
    with _lock:
        if _runs.pop(run_id, None) is None:
            return False
    shutil.rmtree(_run_dir(run_id), ignore_errors=True)
    return True


def _evict_locked() -> None:
    """Drop the oldest finished runs once the store exceeds its cap."""
    if len(_runs) <= MAX_STORED_RUNS:
        return
    finished = sorted(
        (r for r in _runs.values() if r["status"] in {"succeeded", "failed"}),
        key=lambda r: r["created_at"],
    )
    for run in finished[: len(_runs) - MAX_STORED_RUNS]:
        _runs.pop(run["run_id"], None)
        shutil.rmtree(_run_dir(run["run_id"]), ignore_errors=True)


def save_model(run_id: str, pipeline) -> Path:
    directory = _run_dir(run_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MODEL_FILENAME
    joblib.dump(pipeline, path)
    return path


def load_model(run_id: str):
    """Load a fitted pipeline, or None when the run has no usable artifact."""
    path = _run_dir(run_id) / MODEL_FILENAME
    if not path.exists():
        return None
    return joblib.load(path)


def model_exists(run_id: str) -> bool:
    return (_run_dir(run_id) / MODEL_FILENAME).exists()


def reset_for_tests() -> None:
    """Clear both caches. Used by the test suite, never by the app."""
    global _loaded
    with _lock:
        _runs.clear()
        _loaded = False
