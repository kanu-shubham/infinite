"""Runtime configuration loaded from environment.

All knobs that affect benchmark deltas live here so a single env-file
swap reproduces a configuration without code changes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    # --- Model ---
    model: str = os.getenv("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    dtype: str = os.getenv("DTYPE", "bfloat16")
    max_model_len: int = _int("MAX_MODEL_LEN", 8192)
    tensor_parallel_size: int = _int("TENSOR_PARALLEL_SIZE", 1)
    gpu_memory_utilization: float = _float("GPU_MEMORY_UTILIZATION", 0.90)

    # --- Optimizations under test ---
    enable_prefix_caching: bool = _bool("ENABLE_PREFIX_CACHING", False)
    enable_chunked_prefill: bool = _bool("ENABLE_CHUNKED_PREFILL", False)
    max_num_batched_tokens: int = _int("MAX_NUM_BATCHED_TOKENS", 8192)
    max_num_seqs: int = _int("MAX_NUM_SEQS", 256)

    # --- Server / admission ---
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = _int("PORT", 8000)
    # Hard ceiling on concurrently in-flight engine requests. Beyond this we
    # return 429 with Retry-After instead of letting tail latency degrade.
    max_inflight: int = _int("MAX_INFLIGHT", 192)
    # Additional waiters allowed to queue before admission rejects.
    max_queue: int = _int("MAX_QUEUE", 64)
    request_timeout_s: float = _float("REQUEST_TIMEOUT_S", 120.0)
    # Per-request output cap; protects KV budget from runaway generations.
    max_output_tokens: int = _int("MAX_OUTPUT_TOKENS", 1024)

    # --- Observability ---
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
