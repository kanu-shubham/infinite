"""
Rolling latency tracker for p50 / p95 / p99.

Why percentiles matter more than averages
─────────────────────────────────────────
Average latency hides tail behaviour.  A service that responds in 1 ms 99%
of the time but 5 000 ms 1% of the time has a fine average (~51 ms) but is
broken in practice.  SLAs are almost always expressed as percentiles:

    "p99 < 100 ms" means 99 out of 100 requests complete under 100 ms.
    "p999 < 500 ms" means 999 out of 1 000 requests complete under 500 ms.

Tail latency (p99, p999) worsens under load because:
  1. Queue depth grows → requests wait longer before being served.
  2. Thread / connection pool exhaustion → requests block.
  3. GC pauses / lock contention hit the unlucky 1%.

This tracker keeps a 60-second sliding window so metrics stay current.
"""

import threading
import time
from collections import deque
from typing import Optional


class LatencyTracker:
    def __init__(self, window_seconds: float = 60.0) -> None:
        self._window = window_seconds
        # Deque of (monotonic_timestamp, latency_ms) pairs
        self._samples: deque[tuple[float, float]] = deque()
        self._lock = threading.Lock()
        self._total_requests = 0

    def record(self, latency_ms: float) -> None:
        now = time.monotonic()
        with self._lock:
            self._samples.append((now, latency_ms))
            self._total_requests += 1
            self._evict(now)

    def _evict(self, now: float) -> None:
        cutoff = now - self._window
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def stats(self) -> dict:
        now = time.monotonic()
        with self._lock:
            self._evict(now)
            latencies = sorted(s[1] for s in self._samples)
            total = self._total_requests

        n = len(latencies)
        if n == 0:
            return {
                "window_seconds":  self._window,
                "count_in_window": 0,
                "total_requests":  total,
                "rps":             0.0,
                "p50_ms":          None,
                "p95_ms":          None,
                "p99_ms":          None,
                "p999_ms":         None,
                "min_ms":          None,
                "max_ms":          None,
            }

        def pct(p: float) -> Optional[float]:
            idx = min(int(n * p), n - 1)
            return round(latencies[idx], 3)

        return {
            "window_seconds":  self._window,
            "count_in_window": n,
            "total_requests":  total,
            "rps":             round(n / self._window, 1),
            "p50_ms":          pct(0.50),
            "p95_ms":          pct(0.95),
            "p99_ms":          pct(0.99),
            "p999_ms":         pct(0.999) if n >= 1000 else None,
            "min_ms":          round(latencies[0], 3),
            "max_ms":          round(latencies[-1], 3),
        }
