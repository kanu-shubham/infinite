"""Admission control: bounded inflight + bounded wait queue.

Why both limits?

  * `max_inflight` caps work the engine actually runs concurrently. vLLM's
    own scheduler will multiplex these via continuous batching, but the
    KV cache is finite -- once it spills to recompute or preemption, p99
    TTFT collapses. Capping inflight keeps us on the healthy side of the
    saturation cliff.

  * `max_queue` caps how long we'll let a client wait *before* admission.
    Without this, a traffic spike turns into an unbounded backlog and
    every client times out instead of fast-failing the overflow.

Rejected requests get HTTP 429 + Retry-After so well-behaved clients
back off instead of hammering us.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass

from . import metrics


class Overloaded(Exception):
    """Raised when no admission slot is available and queue is full."""

    def __init__(self, retry_after_s: float):
        self.retry_after_s = retry_after_s
        super().__init__(f"overloaded; retry after {retry_after_s:.1f}s")


@dataclass
class Admission:
    max_inflight: int
    max_queue: int

    def __post_init__(self) -> None:
        self._sem = asyncio.Semaphore(self.max_inflight)
        self._waiting = 0
        self._lock = asyncio.Lock()

    async def _try_reserve_queue_slot(self) -> None:
        async with self._lock:
            if self._waiting >= self.max_queue and self._sem.locked():
                # Estimate retry hint from current load; coarse but useful.
                metrics.rejected_total.labels(reason="queue_full").inc()
                raise Overloaded(retry_after_s=1.0)
            self._waiting += 1
            metrics.queue_depth.set(self._waiting)

    async def _release_queue_slot(self) -> None:
        async with self._lock:
            self._waiting = max(0, self._waiting - 1)
            metrics.queue_depth.set(self._waiting)

    @asynccontextmanager
    async def slot(self):
        await self._try_reserve_queue_slot()
        try:
            await self._sem.acquire()
        finally:
            await self._release_queue_slot()
        metrics.inflight.inc()
        try:
            yield
        finally:
            metrics.inflight.dec()
            self._sem.release()
