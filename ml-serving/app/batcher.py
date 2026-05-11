"""
Asyncio micro-batcher for server-side batching.

The idea
─────────
From the client's perspective every request looks like a single prediction.
Under the hood the server groups concurrent requests that arrive within a
short window (max_wait_ms) and calls the model once for the whole group.

The core tradeoff
──────────────────
  Throughput ↑  because fixed inference overhead is amortised over N items.
               At 2 000 RPS and a batch window of 10 ms, 20+ requests land
               per window, so each model call serves ~20 items instead of 1.

  p50 latency ↓ (slightly)  because batched inference is cheaper per item —
               less numpy dispatch overhead per prediction.

  p99 latency ↑  because some requests wait the full max_wait_ms before the
               batch is flushed.  Under light load most windows are size=1
               (timeout fires immediately), so the penalty is ≤ max_wait_ms.

This is why frameworks like TorchServe and Triton expose a `max_batch_delay`
knob.  Tuning it is a dial between throughput and tail latency.

Asyncio design
──────────────
  • The worker loop is a single background Task on the event loop.
  • Each caller awaits a Future that the worker resolves after inference.
  • Inference itself runs in a ThreadPoolExecutor so it does not block the
    event loop (asyncio.get_running_loop().run_in_executor).
  • Thread safety: all Queue / Future operations happen on the event loop
    thread; inference runs in a separate thread but only touches its own
    numpy arrays.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _Item:
    features:    dict
    future:      asyncio.Future
    enqueued_at: float = field(default_factory=time.monotonic)


class MicroBatcher:
    def __init__(
        self,
        model,
        max_batch_size: int   = 32,
        max_wait_ms:    float = 10.0,
    ) -> None:
        self._model         = model
        self._max_batch     = max_batch_size
        self._max_wait      = max_wait_ms / 1_000.0   # convert to seconds
        self._queue:  asyncio.Queue | None = None
        self._task:   asyncio.Task  | None = None
        # Exposed for /metrics
        self.total_batches    = 0
        self.total_items      = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._queue = asyncio.Queue()
        self._task  = asyncio.create_task(self._worker(), name="micro-batcher")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    # ── Public API ────────────────────────────────────────────────────────────

    async def predict(self, features: dict) -> dict:
        """
        Enqueue a single prediction request and await its result.

        The caller sees this as a normal async function call.  Internally it
        sits in the batch queue until either:
          (a) max_batch_size items have accumulated, or
          (b) max_wait_ms has elapsed since the first item in the batch arrived.
        """
        loop   = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        await self._queue.put(_Item(features=features, future=future))
        return await future

    # ── Worker ────────────────────────────────────────────────────────────────

    async def _worker(self) -> None:
        loop = asyncio.get_running_loop()

        while True:
            # Block until the first item of a new batch arrives.
            first: _Item = await self._queue.get()
            batch = [first]
            deadline = first.enqueued_at + self._max_wait

            # Drain the queue: collect more items until the batch is full
            # or the deadline passes.
            while len(batch) < self._max_batch:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                try:
                    item: _Item = await asyncio.wait_for(
                        self._queue.get(), timeout=timeout
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            # Run inference in a thread so the event loop stays responsive.
            hotels = [item.features for item in batch]
            try:
                result = await loop.run_in_executor(
                    None, self._model.predict_batch, hotels
                )
                for idx, item in enumerate(batch):
                    if not item.future.done():
                        item.future.set_result(result["predictions"][idx])
            except Exception as exc:
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(exc)

            self.total_batches += 1
            self.total_items   += len(batch)

    @property
    def avg_batch_size(self) -> float:
        if self.total_batches == 0:
            return 0.0
        return round(self.total_items / self.total_batches, 2)
