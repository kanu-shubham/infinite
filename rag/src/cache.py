"""
Query Result Cache

Identical queries over the same corpus should not hit the LLM twice.
This module provides a simple in-memory TTL cache keyed on
(query, pattern) — the two things that determine a unique response.

Production note
---------------
Replace with Redis for multi-process / multi-server deployments:
  key   = f"rag:{hashlib.sha256(f'{query}:{pattern}'.encode()).hexdigest()}"
  value = json.dumps(response_dict)
  ttl   = 3600
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from src.generation.generator import RAGResponse


@dataclass
class _CacheEntry:
    response: RAGResponse
    expires_at: float


class QueryCache:
    """
    In-memory LRU-ish cache with TTL eviction.

    Parameters
    ----------
    ttl_seconds : how long a cached response stays valid (default 1 hour)
    max_size    : maximum number of entries before oldest are evicted
    """

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000):
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._store: Dict[str, _CacheEntry] = {}
        self.hits = 0
        self.misses = 0

    def _key(self, query: str, pattern: str) -> str:
        raw = f"{query.strip().lower()}::{pattern}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, query: str, pattern: str) -> Optional[RAGResponse]:
        key = self._key(query, pattern)
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None
        if time.time() > entry.expires_at:
            del self._store[key]
            self.misses += 1
            return None
        self.hits += 1
        return entry.response

    def set(self, query: str, pattern: str, response: RAGResponse) -> None:
        # Evict oldest entry if at capacity
        if len(self._store) >= self._max_size:
            oldest_key = next(iter(self._store))
            del self._store[oldest_key]

        key = self._key(query, pattern)
        self._store[key] = _CacheEntry(
            response=response,
            expires_at=time.time() + self._ttl,
        )

    def invalidate(self) -> None:
        """Clear all cached entries (call after re-indexing)."""
        self._store.clear()

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        return {
            "size": len(self._store),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hit_rate:.1%}",
        }
