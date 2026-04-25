"""Compare two benchmark JSONs and print a delta table.

Usage:
    python -m bench.compare --baseline base.json --candidate apc.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _delta(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return "-"
    if a == 0:
        return "n/a"
    pct = (b - a) / a * 100.0
    sign = "+" if pct >= 0 else ""
    return f"{b:.2f} ({sign}{pct:.1f}%)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    args = ap.parse_args()

    base = json.loads(args.baseline.read_text())["summary"]
    cand = json.loads(args.candidate.read_text())["summary"]

    rows = [
        ("ttft_ms.p50", base["ttft_ms"]["p50"], cand["ttft_ms"]["p50"]),
        ("ttft_ms.p90", base["ttft_ms"]["p90"], cand["ttft_ms"]["p90"]),
        ("ttft_ms.p99", base["ttft_ms"]["p99"], cand["ttft_ms"]["p99"]),
        ("itl_ms.p50",  base["itl_ms"]["p50"],  cand["itl_ms"]["p50"]),
        ("itl_ms.p90",  base["itl_ms"]["p90"],  cand["itl_ms"]["p90"]),
        ("itl_ms.p99",  base["itl_ms"]["p99"],  cand["itl_ms"]["p99"]),
        ("e2e_ms.p50",  base["e2e_ms"]["p50"],  cand["e2e_ms"]["p50"]),
        ("e2e_ms.p99",  base["e2e_ms"]["p99"],  cand["e2e_ms"]["p99"]),
        ("throughput_tok_s", base["throughput_tok_s"], cand["throughput_tok_s"]),
        ("requests_per_s", base["requests_per_s"], cand["requests_per_s"]),
    ]

    width = max(len(name) for name, *_ in rows)
    print(f"{'metric':<{width}}  {'baseline':>14}  {'candidate (delta)':>22}")
    print("-" * (width + 2 + 14 + 2 + 22))
    for name, b, c in rows:
        b_s = f"{b:.2f}" if isinstance(b, (int, float)) else "-"
        print(f"{name:<{width}}  {b_s:>14}  {_delta(b, c):>22}")


if __name__ == "__main__":
    main()
