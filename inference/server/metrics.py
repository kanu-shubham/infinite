"""Prometheus metrics for the inference service.

We expose enough to drive the four production SLOs:
  * Availability         -> requests_total{outcome=}
  * TTFT / ITL latency   -> ttft_seconds, itl_seconds histograms
  * Saturation           -> inflight gauge, queue_depth gauge, kv_cache_usage
  * Admission pressure   -> rejected_total{reason=}
"""
from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

REGISTRY = CollectorRegistry(auto_describe=True)

# Latency buckets tuned for an interactive chat SLO: fine resolution
# under 1 s where TTFT lives, coarser past that for tail clipping.
_LATENCY_BUCKETS = (
    0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5,
    0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0,
)
# ITL is sub-second per token; tighter buckets near 0.
_ITL_BUCKETS = (
    0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.2, 0.5, 1.0,
)

requests_total = Counter(
    "inference_requests_total",
    "Completed requests by outcome.",
    ["outcome"],
    registry=REGISTRY,
)
rejected_total = Counter(
    "inference_rejected_total",
    "Requests rejected before reaching the engine.",
    ["reason"],
    registry=REGISTRY,
)
ttft_seconds = Histogram(
    "inference_ttft_seconds",
    "Time to first token, server-measured.",
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
itl_seconds = Histogram(
    "inference_itl_seconds",
    "Inter-token latency, per emitted token after the first.",
    buckets=_ITL_BUCKETS,
    registry=REGISTRY,
)
e2e_seconds = Histogram(
    "inference_e2e_seconds",
    "End-to-end request duration.",
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
output_tokens = Histogram(
    "inference_output_tokens",
    "Output token count per request.",
    buckets=(1, 8, 32, 64, 128, 256, 512, 1024, 2048),
    registry=REGISTRY,
)
inflight = Gauge(
    "inference_inflight",
    "Requests currently being generated.",
    registry=REGISTRY,
)
queue_depth = Gauge(
    "inference_queue_depth",
    "Requests waiting on admission slot.",
    registry=REGISTRY,
)
kv_cache_usage = Gauge(
    "inference_kv_cache_usage_ratio",
    "Fraction of KV cache blocks in use (0..1).",
    registry=REGISTRY,
)


def render() -> tuple[bytes, str]:
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
