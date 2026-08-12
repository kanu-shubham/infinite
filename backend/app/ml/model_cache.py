"""In-process cache of fitted pipelines.

`registry.load_model()` reads a file and unpickles it. That is fine once per
training run and ruinous once per prediction: it is tens of milliseconds of
disk and CPU on a path whose whole budget might be 25ms.

This module holds fitted pipelines in memory instead. It is the single largest
latency win available to the serving path, and the reason it needs care rather
than a bare dict is concurrency: several requests can miss on the same run id
at once, and a model can be replaced while requests are still using the old one.
"""

import threading
from collections import OrderedDict
from typing import Any, Dict, Optional

from ..config import MODEL_CACHE_SIZE
from . import registry

_lock = threading.RLock()
_cache: "OrderedDict[str, Any]" = OrderedDict()
_stats = {"hits": 0, "misses": 0, "evictions": 0, "loads": 0}

# One lock per run id, so two requests missing on *different* models load in
# parallel instead of queueing behind each other.
_load_locks: Dict[str, threading.Lock] = {}


def _load_lock_for(run_id: str) -> threading.Lock:
    with _lock:
        return _load_locks.setdefault(run_id, threading.Lock())


def get(run_id: str) -> Optional[Any]:
    """Return the fitted pipeline for `run_id`, loading it once if needed.

    Load-through with double-checked locking: the fast path takes the cache
    lock only briefly, and a miss serialises per run id so ten concurrent
    requests trigger one disk read rather than ten.
    """
    with _lock:
        if run_id in _cache:
            _cache.move_to_end(run_id)          # LRU: mark as recently used
            _stats["hits"] += 1
            return _cache[run_id]
        _stats["misses"] += 1

    with _load_lock_for(run_id):
        # Re-check: another thread may have loaded it while we waited.
        with _lock:
            if run_id in _cache:
                _cache.move_to_end(run_id)
                return _cache[run_id]

        pipeline = registry.load_model(run_id)
        if pipeline is None:
            return None

        with _lock:
            _stats["loads"] += 1
            _cache[run_id] = pipeline
            _cache.move_to_end(run_id)
            while len(_cache) > MODEL_CACHE_SIZE:
                # Evicting only drops our reference. Any request still holding
                # the old object keeps working until it finishes — Python's
                # refcount frees it afterwards. That is what makes replacing a
                # live model safe.
                _cache.popitem(last=False)
                _stats["evictions"] += 1

        return pipeline


def invalidate(run_id: str) -> bool:
    """Drop one model, e.g. when its run is deleted or the artifact is replaced."""
    with _lock:
        _load_locks.pop(run_id, None)
        return _cache.pop(run_id, None) is not None


def warm(run_ids: list[str]) -> Dict[str, bool]:
    """Preload models at startup so the first real request is not the slow one.

    Without this the first user after every deploy pays the cold-load cost —
    which is exactly the request most likely to be a health check or a canary.
    """
    return {run_id: get(run_id) is not None for run_id in run_ids}


def stats() -> dict:
    with _lock:
        total = _stats["hits"] + _stats["misses"]
        return {
            **_stats,
            "size": len(_cache),
            "capacity": MODEL_CACHE_SIZE,
            "hit_rate": round(_stats["hits"] / total, 4) if total else None,
            "cached_runs": list(_cache.keys()),
        }


def clear() -> None:
    with _lock:
        _cache.clear()
        _load_locks.clear()
        for key in _stats:
            _stats[key] = 0
