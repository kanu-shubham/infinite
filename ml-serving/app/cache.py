"""
In-memory LRU prediction cache.

Why caching is a latency trick
────────────────────────────────
For a ranking model serving hotel search results, many users search for the
same popular destinations with similar filters.  If the model is deterministic
(no stochastic dropout, no per-request context), the same feature vector will
always produce the same output.  Caching that output means:

  cache hit  → ~0.01 ms  (dict lookup)
  cache miss → ~1–5 ms   (full inference)

Cache hit rate depends on input cardinality.  In production you'd see:
  - Very high hit rate for popular searches (London, Paris, etc.)
  - Near-zero for long-tail feature combinations

The distributed equivalent is Redis with a short TTL.  The per-replica
in-memory cache here is simpler but doesn't share state across replicas.

Key design: hash the feature dict, not the raw request, so that different
request_ids with identical hotel features share the same cache entry.
"""

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Optional


class PredictionCache:
    def __init__(self, max_size: int = 10_000, ttl_seconds: float = 300.0) -> None:
        self._max_size = max_size
        self._ttl      = ttl_seconds
        # OrderedDict preserves insertion order → O(1) LRU eviction
        self._store: OrderedDict[str, tuple[dict, float]] = OrderedDict()
        self._lock  = threading.Lock()
        self._hits   = 0
        self._misses = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, features: dict) -> Optional[dict]:
        key = self._hash(features)
        with self._lock:
            if key not in self._store:
                self._misses += 1
                return None
            value, expires_at = self._store[key]
            if time.monotonic() > expires_at:
                del self._store[key]
                self._misses += 1
                return None
            # Move to end (most-recently-used)
            self._store.move_to_end(key)
            self._hits += 1
            return value

    def set(self, features: dict, value: dict) -> None:
        key = self._hash(features)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (value, time.monotonic() + self._ttl)
            # Evict least-recently-used when over capacity
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)

    @property
    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size":     len(self._store),
                "capacity": self._max_size,
                "hits":     self._hits,
                "misses":   self._misses,
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
            }

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _hash(features: dict) -> str:
        payload = json.dumps(features, sort_keys=True, separators=(",", ":"))
        return hashlib.blake2s(payload.encode(), digest_size=16).hexdigest()
