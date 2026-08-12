"""Measure the serving path: where the milliseconds actually go.

    python -m benchmarks.serving_latency

Reports the three things that decide production latency — model loading,
feature lookup, and inference — separately, because they have completely
different fixes and an aggregate number hides which one is hurting.
"""

import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("ML_DATASET_ROWS", "6000")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ml import feature_store, model_cache, registry  # noqa: E402
from app.ml.dataset import feature_columns, load_bookings  # noqa: E402
from app.ml.models import resolve_hyperparameters  # noqa: E402
from app.ml.pipeline import build_pipeline  # noqa: E402

TRIALS = 200


def percentiles(samples):
    ordered = sorted(samples)
    return {
        "p50": round(statistics.median(ordered), 3),
        "p95": round(ordered[int(len(ordered) * 0.95)], 3),
        "p99": round(ordered[int(len(ordered) * 0.99)], 3),
        "max": round(ordered[-1], 3),
    }


def main() -> None:
    registry.RUNS_DIR = Path(tempfile.mkdtemp())
    feature_store.reset_backend_for_tests()

    columns = feature_columns("cancellation")
    features = columns["numeric"] + columns["categorical"]
    frame = load_bookings()

    pipeline = build_pipeline(
        columns["numeric"],
        columns["categorical"],
        "gradient_boosting_classifier",
        resolve_hyperparameters("gradient_boosting_classifier", {}),
        42,
    ).fit(frame[features], frame["is_canceled"])

    run_id = "bench-run"
    registry.save_model(run_id, pipeline)
    artifact = registry.RUNS_DIR / run_id / "model.joblib"
    print(f"model artifact: {artifact.stat().st_size / 1024:.0f} KB\n")

    materialized = feature_store.materialize("cancellation", limit=2000)
    print(f"feature store:  {materialized['backend']}, {materialized['entities']} entities\n")

    entity_ids = [f"BK-{i:06d}" for i in range(1, 2001)]
    single = [entity_ids[0]]

    # ── 1 · Model loading: disk every time vs warm cache ─────────────────────
    cold = []
    for _ in range(20):                       # 20 only — it is slow on purpose
        model_cache.clear()
        start = time.perf_counter()
        registry.load_model(run_id)
        cold.append((time.perf_counter() - start) * 1000)

    model_cache.clear()
    model_cache.get(run_id)                   # prime
    warm = []
    for _ in range(TRIALS):
        start = time.perf_counter()
        model_cache.get(run_id)
        warm.append((time.perf_counter() - start) * 1000)

    print("1 · MODEL LOADING")
    print(f"   from disk each request  {percentiles(cold)}")
    print(f"   warm cache              {percentiles(warm)}")
    print(f"   -> {statistics.median(cold) / max(statistics.median(warm), 1e-6):,.0f}x faster\n")

    # ── 2 · Feature lookup, by batch size ────────────────────────────────────
    print("2 · FEATURE LOOKUP (Redis round trip)")
    for size in (1, 10, 100, 1000):
        batch = entity_ids[:size]
        samples = []
        for _ in range(50):
            start = time.perf_counter()
            feature_store.lookup("cancellation", batch)
            samples.append((time.perf_counter() - start) * 1000)
        per_entity = statistics.median(samples) / size
        print(f"   {size:>5} entities  {percentiles(samples)}  ({per_entity:.4f} ms/entity)")

    # Sequential lookups, to show what batching buys.
    start = time.perf_counter()
    for entity in entity_ids[:100]:
        feature_store.lookup("cancellation", [entity])
    sequential = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    feature_store.lookup("cancellation", entity_ids[:100])
    batched = (time.perf_counter() - start) * 1000
    print(f"   100 one-by-one {sequential:.1f} ms  vs  batched {batched:.1f} ms"
          f"  -> {sequential / batched:.0f}x\n")

    # ── 3 · Inference ────────────────────────────────────────────────────────
    print("3 · INFERENCE (sklearn predict)")
    for size in (1, 10, 100):
        results = feature_store.lookup("cancellation", entity_ids[:size])
        batch_frame = feature_store.to_frame("cancellation", results)
        pipeline.predict(batch_frame)         # warm up numpy/BLAS
        samples = []
        for _ in range(50):
            start = time.perf_counter()
            pipeline.predict(batch_frame)
            samples.append((time.perf_counter() - start) * 1000)
        print(f"   {size:>5} rows      {percentiles(samples)}"
              f"  ({statistics.median(samples) / size:.4f} ms/row)")

    # ── 4 · The whole request ────────────────────────────────────────────────
    print("\n4 · END TO END (lookup + predict, single entity)")
    run = {"run_id": run_id, "status": "succeeded",
           "config": {"target": "cancellation", "task": "classification"}}
    from app.ml import inference

    inference.predict_by_entity(run, single)  # warm
    samples = []
    for _ in range(TRIALS):
        start = time.perf_counter()
        inference.predict_by_entity(run, single)
        samples.append((time.perf_counter() - start) * 1000)
    print(f"   warm cache              {percentiles(samples)}")

    cold_e2e = []
    for _ in range(20):
        model_cache.clear()
        start = time.perf_counter()
        inference.predict_by_entity(run, single)
        cold_e2e.append((time.perf_counter() - start) * 1000)
    print(f"   cold model each request {percentiles(cold_e2e)}")

    print(f"\n   cache stats: {model_cache.stats()}")


if __name__ == "__main__":
    main()
