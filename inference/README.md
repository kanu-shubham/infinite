# vLLM Inference Service

A small but production-shaped inference service: vLLM behind FastAPI, with
admission control, streaming, metrics, and a benchmark harness that
measures **TTFT** (time-to-first-token) and **ITL** (inter-token latency)
end-to-end from a real HTTP client.

The service ships in three configurations so the impact of each
optimization can be measured against a single baseline:

1. **Baseline** — vLLM defaults (continuous batching only).
2. **+APC** — Automatic Prefix Caching enabled.
3. **+APC +Chunked Prefill** — APC plus chunked prefill.

---

## Model

`Qwen/Qwen2.5-1.5B-Instruct` (bf16). Picked because it:

- fits comfortably on a single 24 GB GPU at `max_model_len=8192` so the
  knobs we toggle, not memory pressure, drive the deltas;
- has a permissive license suitable for redistribution;
- is small enough that prefill cost is observable but not so large that
  ITL is dominated by raw FLOPs (so optimization wins are visible).

Swap `MODEL` in the env-file to test a larger model — nothing else
changes.

---

## Why these two optimizations (and not the others)

The brief asked for production-grade considerations along the lines of
how Claude handles many concurrent requests. The two optimizations below
were chosen because they target the two problems any large multi-tenant
inference fleet hits first:

### 1. Automatic Prefix Caching (APC)

Real traffic is dominated by **shared prefixes**: a system prompt, a
tool definition block, a RAG header, few-shot exemplars. Without prefix
caching, every request re-runs prefill over those identical tokens.

vLLM's APC hashes prompt prefix blocks and reuses the KV blocks across
requests when the prefix matches. This is the same principle as Claude's
prompt caching — at the infrastructure layer rather than billed to the
user. The win is concentrated on TTFT, because most of the prefill work
disappears.

Enabled with `--enable-prefix-caching` (or `ENABLE_PREFIX_CACHING=true`
in our env-file).

### 2. Chunked Prefill

The default scheduler runs a request's prefill as a single step. While
that step runs, every other request's decode pauses. With long prompts,
this shows up as **TTFT and ITL spikes for everyone else** the moment
one big prompt hits the queue.

Chunked prefill slices a long prefill into smaller chunks and interleaves
them with decode steps of already-running requests. The result: smoother
ITL tails under mixed load, at a small cost to single-request prefill
throughput.

Enabled with `--enable-chunked-prefill` plus a tighter
`max_num_batched_tokens` (we use 2048).

### Optimizations we deliberately did NOT pick

- **Quantization (AWQ/FP8)** — orthogonal win, but changes accuracy and
  conflates the comparison. Worth doing in a real deployment; out of
  scope here.
- **Speculative decoding** — large ITL win but hardware/model dependent
  and fiddly to tune; not a fair "two-line config" optimization.
- **Tensor parallelism** — only useful when one GPU isn't enough. Not
  the bottleneck for a 1.5B model.

---

## Methodology

- Two workloads, both warmed up (8 requests) before sampling:
  - `shared_system`: every request shares a ~1k-token system prompt.
    Designed to expose APC.
  - `mixed`: 30% prefill-heavy + 70% decode-heavy, fired together.
    Designed to expose chunked prefill.
- Concurrency: 16 (shared_system), 24 (mixed). Held constant across runs.
- 200 measured requests per run. `temperature=0.0` for determinism.
- TTFT and ITL are measured **on the client**, from the HTTP request
  send through each SSE delta arrival. Server-side histograms exist too
  (`/metrics`) but the client view is the one that matches user
  experience.
- Same hardware, same vLLM build, same model weights across runs.
- Engine flags are immutable after boot, so the server is restarted
  between runs.

Reproduce:

```bash
make install
# Terminal A: pick one config and start the server
make server-baseline                       # then ^C, switch, repeat
# Terminal B: run the benchmark for the matching label
make bench-baseline
make bench-apc
make bench-apc-chunked
make compare
```

Outputs land in `bench/results/`.

---

## Results

> The numbers below are illustrative deltas based on published vLLM
> benchmarks for a 1–2B model on a single A10/A100. The harness
> (`bench/benchmark.py`) and comparator (`bench/compare.py`) produce
> real numbers on your hardware — replace this table with your own
> output. The shape (direction and rough magnitude) is what to expect.

### Shared-system workload — baseline → +APC

| Metric            | Baseline | +APC          | Delta     |
| ----------------- | -------: | ------------: | --------: |
| TTFT p50 (ms)     |      210 |            55 |  **−74%** |
| TTFT p90 (ms)     |      340 |            95 |  **−72%** |
| TTFT p99 (ms)     |      520 |           180 |  **−65%** |
| ITL p50 (ms)      |       22 |            22 |     ≈0%   |
| ITL p99 (ms)      |       55 |            48 |    −13%   |
| Throughput (tok/s)|      980 |          1380 |  **+41%** |

Reading: APC erases prefill cost for the shared 1k-token system prompt,
so TTFT collapses. ITL barely moves because decode wasn't the
bottleneck. Throughput rises because GPU time previously spent on
redundant prefill is now spent on decode.

### Mixed workload — +APC → +APC +Chunked Prefill

| Metric            |   +APC | +APC +Chunked  | Delta     |
| ----------------- | -----: | -------------: | --------: |
| TTFT p50 (ms)     |     90 |            110 |    +22%   |
| TTFT p99 (ms)     |    760 |            290 |  **−62%** |
| ITL p50 (ms)      |     22 |             24 |    +9%    |
| ITL p99 (ms)      |    140 |             52 |  **−63%** |
| Throughput (tok/s)|   1320 |           1280 |    −3%    |

Reading: chunked prefill trades a small p50 regression (every step now
contains a smaller batch) for a much tighter tail. The *mean* user got
slightly slower; the *worst* user got dramatically faster. That's
exactly the tradeoff you want for an interactive SLO.

---

## Production-grade considerations

The handler in `server/main.py` is small on purpose, but everything
around it is shaped for multi-tenant traffic. Each item below is a real
problem fleets like Claude's hit; the linked code shows where it's
addressed.

- **Admission control with two limits**
  (`server/admission.py`). `max_inflight` caps concurrent engine work so
  KV cache stays on the healthy side of the saturation cliff;
  `max_queue` caps how long we'll let clients wait before fast-failing
  with HTTP 429 + `Retry-After`. Without the second limit, a traffic
  spike turns into an unbounded backlog and every client times out.

- **Client-disconnect → engine abort**
  (`server/engine.py:Engine.stream`). When the SSE peer disappears,
  cancellation propagates to `engine.abort(request_id)`, freeing the KV
  blocks immediately. Without this, the engine generates to
  `max_tokens` for a client that's already gone — pure waste, and a
  textbook way to make tail latency worse on the next spike.

- **Per-step timeout, not per-request timeout**
  (`server/main.py:_timed`). A long but healthy generation should not
  be killed; a hung step should. Wrapping the whole stream in a single
  timeout punishes legitimate long answers.

- **Liveness vs. readiness split** (`/healthz` vs. `/readyz`).
  Loading a multi-billion-parameter model takes minutes. A combined
  health endpoint gets the pod killed by the kubelet during startup;
  the split keeps liveness cheap and lets the LB hold traffic until
  the engine is actually ready.

- **Prometheus metrics shaped for SLOs** (`server/metrics.py`).
  Four buckets: availability (`requests_total{outcome=}`), latency
  (`ttft_seconds`, `itl_seconds` histograms), saturation (`inflight`,
  `queue_depth`, `kv_cache_usage_ratio`), pressure
  (`rejected_total{reason=}`). Latency buckets are tuned for an
  interactive chat SLO — fine resolution under 1 s, coarser past that.

- **Structured JSON logs with request id**
  (`server/logging_setup.py`). One line per request, indexable on
  `request_id`, `outcome`, `ttft_ms`. No regex needed in
  Loki / Cloud Logging.

- **Per-request token cap** (`MAX_OUTPUT_TOKENS`). Prevents one
  runaway generation from owning a KV slot for minutes.

- **Engine flags are immutable, so the env-file is the contract.**
  Three configurations (`configs/*.env`) — flip one, restart, run the
  bench. No code changes between runs, which is the only way deltas
  are trustworthy.

- **Stream backpressure is honoured automatically.** FastAPI's
  `StreamingResponse` awaits each chunk; if the client TCP buffer
  fills, the generator suspends and decode naturally throttles. We
  don't buffer the whole response in memory.

- **Graceful shutdown** (`lifespan`). On SIGTERM, the lifespan's
  `finally` runs `engine.stop()` after FastAPI drains in-flight
  requests, so a rolling deploy doesn't 502.

### Things you'd add for a real fleet (and where they'd plug in)

- **Authn/authz + per-tenant quotas**: middleware in front of
  `/v1/chat`, keyed off the API key. The admission slot can be
  per-tenant rather than global so a single noisy tenant can't starve
  the rest.
- **Token-bucket rate limiting on input + output tokens**, not just
  request count — LLM cost is dominated by tokens, not RPS.
- **Multi-replica routing with prefix-affinity**: route requests with
  the same prompt prefix to the same replica so APC actually hits.
  Without affinity, APC hit rate degrades linearly with replica count.
- **KV-cache-aware autoscaler**: scale on `kv_cache_usage_ratio`
  rather than CPU/GPU util. KV pressure is the actual saturation
  signal; CPU is misleading.
- **Speculative decoding** with a small draft model — orthogonal ITL
  win on top of these two optimizations.
- **Request shadowing / canary** for safe model rollouts.

---

## Layout

```
inference/
├── README.md
├── requirements.txt
├── Dockerfile
├── Makefile
├── server/
│   ├── main.py            # FastAPI app + lifespan + SSE handler
│   ├── engine.py          # AsyncLLMEngine wrapper + KV stats poller
│   ├── admission.py       # bounded inflight + bounded queue
│   ├── config.py          # env-driven Settings
│   ├── metrics.py         # Prometheus collectors
│   ├── logging_setup.py   # JSON logs
│   └── schemas.py         # Pydantic request models
├── bench/
│   ├── benchmark.py       # async TTFT/ITL load generator
│   ├── workloads.py       # shared_system + mixed presets
│   └── compare.py         # diff two run JSONs
├── configs/
│   ├── baseline.env
│   ├── apc.env
│   └── apc_chunked.env
└── scripts/
    ├── run_server.sh
    └── run_bench.sh
```
