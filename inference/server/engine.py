"""Thin wrapper around vLLM's AsyncLLMEngine.

Responsibilities:
  * Build the engine from `Settings` (the only place vLLM args are set).
  * Render chat messages with the model's tokenizer chat template.
  * Stream deltas with monotonic timestamps so the caller can compute
    TTFT / ITL exactly once at the boundary (no double accounting).
  * Surface a periodic KV-cache utilization gauge so we can correlate
    latency regressions with cache pressure.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

from . import metrics
from .config import Settings

log = logging.getLogger(__name__)


@dataclass
class TokenDelta:
    text: str          # incremental text since last delta
    index: int         # 0-based token index within the response
    monotonic_s: float # time.monotonic() at emission


class Engine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._engine: AsyncLLMEngine | None = None
        self._stats_task: asyncio.Task | None = None

    async def start(self) -> None:
        args = AsyncEngineArgs(
            model=self.settings.model,
            dtype=self.settings.dtype,
            max_model_len=self.settings.max_model_len,
            tensor_parallel_size=self.settings.tensor_parallel_size,
            gpu_memory_utilization=self.settings.gpu_memory_utilization,
            enable_prefix_caching=self.settings.enable_prefix_caching,
            enable_chunked_prefill=self.settings.enable_chunked_prefill,
            max_num_batched_tokens=self.settings.max_num_batched_tokens,
            max_num_seqs=self.settings.max_num_seqs,
            disable_log_requests=True,
        )
        log.info(
            "engine starting",
            extra={
                "model": self.settings.model,
                "enable_prefix_caching": self.settings.enable_prefix_caching,
                "enable_chunked_prefill": self.settings.enable_chunked_prefill,
                "max_num_batched_tokens": self.settings.max_num_batched_tokens,
                "max_num_seqs": self.settings.max_num_seqs,
            },
        )
        self._engine = AsyncLLMEngine.from_engine_args(args)
        # Warm the tokenizer so the first user request doesn't pay the import.
        await self._engine.get_tokenizer()
        self._stats_task = asyncio.create_task(self._poll_stats())

    async def stop(self) -> None:
        if self._stats_task:
            self._stats_task.cancel()
        # AsyncLLMEngine has no explicit shutdown in 0.6.x; drop the ref so
        # the background loop in vLLM is GC'd at process exit.
        self._engine = None

    @property
    def ready(self) -> bool:
        return self._engine is not None

    async def _poll_stats(self) -> None:
        """Best-effort KV-cache usage gauge.

        vLLM exposes scheduler stats internally; we read the GPU cache
        usage from the engine's stat logger if available. Falls back to
        a no-op if the private API drifts -- we don't want a metric
        scrape to crash the server.
        """
        assert self._engine is not None
        while True:
            try:
                await asyncio.sleep(1.0)
                stats = getattr(self._engine.engine, "stat_loggers", None)
                if not stats:
                    continue
                for logger in stats.values():
                    usage = getattr(logger, "last_local_stats", None)
                    if usage is None:
                        continue
                    gpu = getattr(usage, "gpu_cache_usage_sys", None)
                    if gpu is not None:
                        metrics.kv_cache_usage.set(float(gpu))
            except asyncio.CancelledError:
                return
            except Exception:  # noqa: BLE001
                # Telemetry must never take down the data plane.
                log.debug("kv stats poll failed", exc_info=True)

    async def _render_prompt(self, messages: list[dict]) -> str:
        assert self._engine is not None
        tok = await self._engine.get_tokenizer()
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    async def stream(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None,
        request_id: str | None = None,
    ) -> AsyncIterator[TokenDelta]:
        assert self._engine is not None, "engine not started"
        rid = request_id or f"req-{uuid.uuid4().hex[:12]}"
        prompt = await self._render_prompt(messages)
        sp = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )

        last_text = ""
        index = 0
        results = self._engine.generate(prompt, sp, request_id=rid)
        try:
            async for out in results:
                # vLLM streams cumulative text per output; emit the diff.
                if not out.outputs:
                    continue
                cur = out.outputs[0].text
                if len(cur) <= len(last_text):
                    continue
                delta = cur[len(last_text):]
                last_text = cur
                yield TokenDelta(
                    text=delta, index=index, monotonic_s=time.monotonic()
                )
                index += 1
        except asyncio.CancelledError:
            # Client disconnect: tell the engine so it frees the KV slot
            # immediately instead of generating to max_tokens.
            try:
                await self._engine.abort(rid)
            finally:
                raise
