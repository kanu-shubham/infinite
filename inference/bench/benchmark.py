"""Async load generator that measures TTFT and ITL from the client side.

Why client-side measurement?
  Server histograms exclude network and SSE-flush time, so they understate
  what users feel. We measure from the moment the request body is sent
  until each SSE delta arrives. This catches everything an end user pays.

Output:
  A JSON file with per-request samples plus aggregate percentiles, ready
  for `bench/compare.py` to diff two runs.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
import orjson

from . import workloads


@dataclass
class Sample:
    ttft_s: float | None = None
    itls_s: list[float] = field(default_factory=list)
    e2e_s: float = 0.0
    tokens: int = 0
    status: str = "ok"


async def _one(client: httpx.AsyncClient, url: str, prompt) -> Sample:
    s = Sample()
    started = time.monotonic()
    body = {
        "messages": prompt.messages,
        "max_tokens": prompt.max_tokens,
        "temperature": 0.0,  # deterministic for fair comparison
        "top_p": 1.0,
        "stream": True,
    }
    last_t: float | None = None
    try:
        async with client.stream(
            "POST", url, json=body, timeout=httpx.Timeout(120.0, connect=10.0)
        ) as resp:
            if resp.status_code == 429:
                s.status = "rejected"
                return s
            if resp.status_code != 200:
                s.status = f"http_{resp.status_code}"
                return s
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                evt = orjson.loads(line[6:])
                if evt.get("type") == "delta":
                    now = time.monotonic()
                    if s.ttft_s is None:
                        s.ttft_s = now - started
                    else:
                        assert last_t is not None
                        s.itls_s.append(now - last_t)
                    last_t = now
                    s.tokens += 1
                elif evt.get("type") == "done":
                    break
                elif evt.get("type") == "error":
                    s.status = f"error:{evt.get('error', 'unknown')}"
                    break
    except (httpx.HTTPError, asyncio.TimeoutError) as e:
        s.status = f"client_error:{type(e).__name__}"
    s.e2e_s = time.monotonic() - started
    return s


async def _worker(client, url, queue: asyncio.Queue, out: list[Sample]):
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return
        out.append(await _one(client, url, item))
        queue.task_done()


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


def aggregate(samples: list[Sample]) -> dict:
    ok = [s for s in samples if s.status == "ok"]
    ttfts = [s.ttft_s for s in ok if s.ttft_s is not None]
    itls = [x for s in ok for x in s.itls_s]
    e2es = [s.e2e_s for s in ok]
    return {
        "n_total": len(samples),
        "n_ok": len(ok),
        "n_rejected": sum(1 for s in samples if s.status == "rejected"),
        "n_error": sum(1 for s in samples if s.status not in {"ok", "rejected"}),
        "ttft_ms": {
            "p50": round(_pct(ttfts, 50) * 1000, 2) if ttfts else None,
            "p90": round(_pct(ttfts, 90) * 1000, 2) if ttfts else None,
            "p99": round(_pct(ttfts, 99) * 1000, 2) if ttfts else None,
            "mean": round(statistics.mean(ttfts) * 1000, 2) if ttfts else None,
        },
        "itl_ms": {
            "p50": round(_pct(itls, 50) * 1000, 2) if itls else None,
            "p90": round(_pct(itls, 90) * 1000, 2) if itls else None,
            "p99": round(_pct(itls, 99) * 1000, 2) if itls else None,
            "mean": round(statistics.mean(itls) * 1000, 2) if itls else None,
        },
        "e2e_ms": {
            "p50": round(_pct(e2es, 50) * 1000, 2) if e2es else None,
            "p99": round(_pct(e2es, 99) * 1000, 2) if e2es else None,
        },
        "throughput_tok_s": (
            round(sum(s.tokens for s in ok) / max(sum(e2es), 1e-9), 2)
            if ok else None
        ),
    }


async def run(
    *,
    url: str,
    workload: str,
    n: int,
    concurrency: int,
    warmup: int,
    seed: int,
) -> dict:
    prompts = workloads.build(workload, n + warmup, seed=seed)
    samples: list[Sample] = []
    limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(http2=False, limits=limits) as client:
        # Warmup: same workload, results discarded. This populates the
        # prefix cache (when enabled) and lets the JIT settle.
        if warmup:
            wq: asyncio.Queue = asyncio.Queue()
            for p in prompts[:warmup]:
                wq.put_nowait(p)
            for _ in range(concurrency):
                wq.put_nowait(None)
            wsink: list[Sample] = []
            await asyncio.gather(*[
                _worker(client, url, wq, wsink) for _ in range(concurrency)
            ])

        q: asyncio.Queue = asyncio.Queue()
        for p in prompts[warmup:]:
            q.put_nowait(p)
        for _ in range(concurrency):
            q.put_nowait(None)
        t0 = time.monotonic()
        await asyncio.gather(*[
            _worker(client, url, q, samples) for _ in range(concurrency)
        ])
        wall = time.monotonic() - t0

    agg = aggregate(samples)
    agg["wall_s"] = round(wall, 3)
    agg["concurrency"] = concurrency
    agg["workload"] = workload
    agg["requests_per_s"] = round(len(samples) / wall, 2) if wall > 0 else None
    return {"summary": agg, "samples": [asdict(s) for s in samples]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1/chat")
    ap.add_argument("--workload", choices=["shared_system", "mixed"], default="shared_system")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default="run")
    args = ap.parse_args()

    result = asyncio.run(run(
        url=args.url,
        workload=args.workload,
        n=args.n,
        concurrency=args.concurrency,
        warmup=args.warmup,
        seed=args.seed,
    ))
    result["label"] = args.label
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(orjson.dumps(result, option=orjson.OPT_INDENT_2))
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
