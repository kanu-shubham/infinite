"""Online feature store.

At training time features are read from a table, in bulk, with all the time in
the world. At serving time a request arrives carrying only an identifier —
`{"booking_id": "BK-003347"}` — and the features have to be found in
single-digit milliseconds. That is what this module is for.

Four questions decide whether a feature store is real or a cache with a nice
name, and each one is answered explicitly below:

1. **How do features get in?**  `materialize()` — a batch job that reads the
   same columns training reads and writes them keyed by entity.
2. **How stale are they?**  Every row carries `materialized_at`; every read
   reports `staleness_seconds`. The TTL is a separate, harder bound.
3. **What happens on a miss?**  Nothing silently. A miss is reported per
   entity, and the caller chooses whether to impute or reject.
4. **How does it stay consistent with training?**  Each row stores a hash of
   the feature-name set it was written with. A read against a model expecting
   a different set fails loudly instead of scoring garbage.

Point 4 is the one that matters. Online/offline skew is the most expensive bug
in production ML precisely because nothing crashes — the model just quietly
gets worse.
"""

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from ..config import FEATURE_MAX_STALENESS_SECONDS, FEATURE_TTL_SECONDS, REDIS_URL
from .dataset import feature_columns, load_bookings

ENTITY_COLUMN = "booking_id"
KEY_PREFIX = "feat"


class FeatureStoreError(RuntimeError):
    """Raised when the store cannot serve a request safely."""


# ── Schema fingerprint ───────────────────────────────────────────────────────
def schema_hash(feature_names: Iterable[str]) -> str:
    """Stable fingerprint of a feature-name set.

    Sorted, so column *order* does not change the hash — only membership does.
    Written alongside every row and checked on every read; that check is the
    whole defence against serving a model features it was not trained on.
    """
    digest = hashlib.sha256("|".join(sorted(feature_names)).encode()).hexdigest()
    return digest[:16]


def _key(namespace: str, entity_id: str) -> str:
    return f"{KEY_PREFIX}:{namespace}:{entity_id}"


# ── Backends ─────────────────────────────────────────────────────────────────
class InMemoryBackend:
    """Per-process dict with TTL. Tests and single-node local runs only.

    Deliberately not a fallback you would ship: with two API replicas each
    holds its own copy, so a feature written by one is invisible to the other.
    """

    name = "memory"

    def __init__(self) -> None:
        self._data: Dict[str, tuple[str, float]] = {}
        self._lock = threading.RLock()

    def mset(self, items: Dict[str, str], ttl: int) -> None:
        expires = time.time() + ttl
        with self._lock:
            for key, value in items.items():
                self._data[key] = (value, expires)

    def mget(self, keys: List[str]) -> List[Optional[str]]:
        now = time.time()
        with self._lock:
            out = []
            for key in keys:
                entry = self._data.get(key)
                if entry is None or entry[1] < now:
                    out.append(None)
                else:
                    out.append(entry[0])
            return out

    def count(self, namespace: str) -> int:
        now = time.time()
        prefix = f"{KEY_PREFIX}:{namespace}:"
        with self._lock:
            return sum(
                1 for k, (_, exp) in self._data.items() if k.startswith(prefix) and exp >= now
            )

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


class RedisBackend:
    """Redis-backed store. One round trip per batch, whatever the batch size."""

    name = "redis"

    def __init__(self, client) -> None:
        self._client = client

    def mset(self, items: Dict[str, str], ttl: int) -> None:
        # A pipeline batches the writes into a single network round trip.
        # SET-with-TTL per key rather than MSET, because MSET cannot set TTLs.
        pipe = self._client.pipeline(transaction=False)
        for key, value in items.items():
            pipe.set(key, value, ex=ttl)
        pipe.execute()

    def mget(self, keys: List[str]) -> List[Optional[str]]:
        if not keys:
            return []
        # MGET is the reason lookup latency does not grow with batch size.
        raw = self._client.mget(keys)
        return [v.decode() if isinstance(v, bytes) else v for v in raw]

    def count(self, namespace: str) -> int:
        # SCAN, never KEYS — KEYS blocks the single-threaded server.
        total = 0
        for _ in self._client.scan_iter(match=f"{KEY_PREFIX}:{namespace}:*", count=1000):
            total += 1
        return total

    def clear(self) -> None:
        for key in self._client.scan_iter(match=f"{KEY_PREFIX}:*", count=1000):
            self._client.delete(key)


_backend = None
_backend_lock = threading.Lock()


def backend():
    """Resolve the backend once. Falls back to memory if Redis is unreachable."""
    global _backend
    if _backend is not None:
        return _backend

    with _backend_lock:
        if _backend is not None:
            return _backend

        if REDIS_URL:
            try:
                import redis

                client = redis.Redis.from_url(
                    REDIS_URL, socket_timeout=0.25, socket_connect_timeout=0.25
                )
                client.ping()
                _backend = RedisBackend(client)
                return _backend
            except Exception:
                # No Redis: keep serving rather than failing to boot. The
                # backend name is reported on every response so the degradation
                # is visible instead of silent.
                pass

        _backend = InMemoryBackend()
        return _backend


def reset_backend_for_tests() -> None:
    global _backend
    with _backend_lock:
        if _backend is not None:
            _backend.clear()
        _backend = None


# ── Materialisation (the write path) ─────────────────────────────────────────
def materialize(target: str, limit: Optional[int] = None, ttl: int = FEATURE_TTL_SECONDS) -> dict:
    """Push feature rows into the online store, keyed by booking id.

    This is the batch job an orchestrator would run on a schedule. It reads
    through `feature_columns(target)` — the *same* function training uses — so
    the two cannot drift apart by construction.
    """
    columns = feature_columns(target)
    feature_names = columns["numeric"] + columns["categorical"]
    fingerprint = schema_hash(feature_names)

    frame = load_bookings()
    if limit:
        frame = frame.head(limit)

    now = time.time()
    payloads: Dict[str, str] = {}
    for row in frame[[ENTITY_COLUMN] + feature_names].itertuples(index=False):
        record = dict(zip([ENTITY_COLUMN] + feature_names, row))
        entity_id = record.pop(ENTITY_COLUMN)
        payloads[_key(target, entity_id)] = json.dumps(
            {
                "f": {k: (None if pd.isna(v) else v) for k, v in record.items()},
                "schema": fingerprint,
                "materialized_at": now,
            },
            default=str,
        )

    store = backend()
    store.mset(payloads, ttl)

    return {
        "target": target,
        "backend": store.name,
        "entities": len(payloads),
        "features_per_entity": len(feature_names),
        "schema_hash": fingerprint,
        "ttl_seconds": ttl,
    }


# ── Lookup (the read path) ───────────────────────────────────────────────────
@dataclass
class LookupResult:
    """One entity's features plus everything the caller needs to judge them."""

    entity_id: str
    features: Dict[str, Any]
    found: bool
    staleness_seconds: Optional[float] = None
    missing_features: List[str] = field(default_factory=list)


def lookup(target: str, entity_ids: List[str]) -> List[LookupResult]:
    """Fetch features for many entities in one round trip.

    Batching matters: N sequential GETs is N network round trips, and at a
    5ms budget you get roughly one.
    """
    columns = feature_columns(target)
    feature_names = columns["numeric"] + columns["categorical"]
    expected_hash = schema_hash(feature_names)

    raw_values = backend().mget([_key(target, e) for e in entity_ids])
    now = time.time()
    results: List[LookupResult] = []

    for entity_id, raw in zip(entity_ids, raw_values):
        if raw is None:
            results.append(LookupResult(entity_id=entity_id, features={}, found=False))
            continue

        payload = json.loads(raw)

        # ── The parity check ──────────────────────────────────────────────
        # The stored row was written for a different feature set than the
        # model expects. Serving it would silently mis-align columns, so this
        # is a hard failure rather than a warning.
        if payload.get("schema") != expected_hash:
            raise FeatureStoreError(
                f"Feature schema mismatch for '{entity_id}': store has "
                f"{payload.get('schema')}, model expects {expected_hash}. "
                "Re-materialise the feature store for this target."
            )

        age = round(now - float(payload["materialized_at"]), 3)
        if FEATURE_MAX_STALENESS_SECONDS and age > FEATURE_MAX_STALENESS_SECONDS:
            raise FeatureStoreError(
                f"Features for '{entity_id}' are {age:.0f}s old, above the "
                f"{FEATURE_MAX_STALENESS_SECONDS}s limit."
            )

        values = payload["f"]
        results.append(
            LookupResult(
                entity_id=entity_id,
                features=values,
                found=True,
                staleness_seconds=age,
                # A null in the store is a real gap, not an absent key. The
                # pipeline's imputers will fill it — same as at training time.
                missing_features=[n for n in feature_names if values.get(n) is None],
            )
        )

    return results


def to_frame(target: str, results: List[LookupResult]) -> pd.DataFrame:
    """Turn lookup results into the exact column set the pipeline expects.

    A total miss becomes a row of NaN rather than a dropped row, so the caller
    still gets one prediction per requested entity and can decide what a
    fully-imputed answer is worth.
    """
    columns = feature_columns(target)
    expected = columns["numeric"] + columns["categorical"]

    rows = [
        {name: result.features.get(name, np.nan) for name in expected} for result in results
    ]
    frame = pd.DataFrame(rows, columns=expected)
    for column in columns["numeric"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def stats(target: str) -> dict:
    """What the store currently holds — for a health endpoint or dashboard."""
    columns = feature_columns(target)
    feature_names = columns["numeric"] + columns["categorical"]
    store = backend()
    return {
        "backend": store.name,
        "target": target,
        "entities": store.count(target),
        "expected_schema_hash": schema_hash(feature_names),
        "ttl_seconds": FEATURE_TTL_SECONDS,
        "max_staleness_seconds": FEATURE_MAX_STALENESS_SECONDS or None,
    }
