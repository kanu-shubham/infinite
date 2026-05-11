"""Closed-loop load generator with target RPS.

Usage:
    python -m ml.benchmarks.load_test --rps 2000 --duration 30 --candidates 50

Reports p50/p95/p99/p999 latencies (client-side and server-reported) and
achieved RPS. The generator is async — a single process can typically saturate
a serving replica at >2000 RPS over loopback.
"""
from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import time

import aiohttp


def _build_payload(n: int) -> dict:
    return {
        "request_id": f"bench-{random.randrange(10**9)}",
        "user": {"user_id": f"u_{random.randrange(50_000)}", "country": "US", "device": "mobile"},
        "context": {
            "destination": random.choice(["PAR", "NYC", "LON", "TYO"]),
            "lead_time_days": random.randint(1, 60),
            "los": random.randint(1, 7),
            "pax": random.randint(1, 4),
            "hour": random.randint(0, 23),
            "dow": random.randint(0, 6),
        },
        "candidates": [
            {"ad_id": f"ad_{random.randrange(20_000)}",
             "advertiser_id": f"adv_{random.randrange(500)}",
             "bid_cpc": round(random.uniform(0.2, 3.0), 3)}
            for _ in range(n)
        ],
    }


async def _worker(session: aiohttp.ClientSession, url: str, queue: asyncio.Queue, lat_client: list, lat_server: list, errs: list, n_candidates: int):
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return
        payload = _build_payload(n_candidates)
        t0 = time.perf_counter_ns()
        try:
            async with session.post(url, json=payload) as r:
                data = await r.json()
                t1 = time.perf_counter_ns()
                if r.status == 200:
                    lat_client.append((t1 - t0) / 1e6)
                    lat_server.append(data.get("latency_us", 0) / 1000.0)
                else:
                    errs.append(r.status)
        except Exception as e:  # network errors counted
            errs.append(repr(e))
        finally:
            queue.task_done()


async def _producer(queue: asyncio.Queue, rps: int, duration: float):
    interval = 1.0 / rps
    end = time.perf_counter() + duration
    next_tick = time.perf_counter()
    while time.perf_counter() < end:
        await queue.put(1)
        next_tick += interval
        sleep_for = next_tick - time.perf_counter()
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1)))))
    return xs[k]


async def _run(args) -> None:
    url = f"{args.host.rstrip('/')}/v1/rank"
    queue: asyncio.Queue = asyncio.Queue(maxsize=args.concurrency * 4)
    lat_client: list[float] = []
    lat_server: list[float] = []
    errs: list = []

    timeout = aiohttp.ClientTimeout(total=2.0)
    conn = aiohttp.TCPConnector(limit=args.concurrency, ttl_dns_cache=300)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        workers = [
            asyncio.create_task(_worker(session, url, queue, lat_client, lat_server, errs, args.candidates))
            for _ in range(args.concurrency)
        ]
        t0 = time.perf_counter()
        await _producer(queue, args.rps, args.duration)
        await queue.join()
        elapsed = time.perf_counter() - t0
        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers)

    n_ok = len(lat_client)
    print(f"\n=== TravelAds rank /v1/rank load test ===")
    print(f"target rps      : {args.rps}")
    print(f"candidates/req  : {args.candidates}")
    print(f"duration        : {elapsed:.1f}s")
    print(f"requests ok     : {n_ok}")
    print(f"errors          : {len(errs)}")
    print(f"achieved rps    : {n_ok / elapsed:.1f}")
    if lat_client:
        print(f"client latency (ms): p50={_pct(lat_client,50):.2f}  p95={_pct(lat_client,95):.2f}  "
              f"p99={_pct(lat_client,99):.2f}  p999={_pct(lat_client,99.9):.2f}  "
              f"mean={statistics.fmean(lat_client):.2f}")
        print(f"server latency (ms): p50={_pct(lat_server,50):.2f}  p95={_pct(lat_server,95):.2f}  "
              f"p99={_pct(lat_server,99):.2f}  p999={_pct(lat_server,99.9):.2f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="http://127.0.0.1:8080")
    p.add_argument("--rps", type=int, default=2000)
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--candidates", type=int, default=50)
    p.add_argument("--concurrency", type=int, default=128)
    args = p.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
