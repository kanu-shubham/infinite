"""
Locust load test for the Hotel Ranker service.

How to run
──────────
  # Web UI (recommended for the learning exercise)
  locust -f locustfile.py --host http://localhost:8000

  # Headless — useful for CI or scripted ramp tests
  locust -f locustfile.py --host http://localhost:8000 \
         --users 200 --spawn-rate 10 --run-time 60s --headless

  # Test one strategy at a time by naming the user class
  locust -f locustfile.py BaselineUser     --host http://localhost:8000
  locust -f locustfile.py MicroBatchedUser --host http://localhost:8000

What to watch
─────────────
Open http://localhost:8089, set Users = 200, Spawn rate = 10.
Switch to the "Charts" tab and observe:

  1. p50 is usually stable — most requests finish quickly.
  2. p99 climbs as concurrency increases — the unlucky 1% waits in a queue.
  3. Throughput (RPS) plateaus — you've found the saturation point.
  4. Failures appear — threadpool / connection pool exhaustion.

Then compare:
  • BaselineUser      → raw inference, no tricks
  • CachedUser        → cache warms up; watch p99 collapse after ~30 s
  • MicroBatchedUser  → higher concurrency, watch avg_batch_size grow at /metrics

Key mental model
─────────────────
  Latency  = time a single request takes end-to-end
  Throughput = requests per second the system can sustain

  They are related by Little's Law:
      L = λ × W
      (mean queue depth = arrival rate × mean wait time)

  As λ (RPS) approaches system capacity:
    • W (latency) grows rapidly
    • p99 diverges from p50 because the tail requests hit a full queue

  Batching increases effective capacity (λ_max) by making each model call
  serve more items.  Caching reduces λ by serving some requests without
  touching the model at all.
"""

import random

from locust import HttpUser, between, task


# ── Feature generators ────────────────────────────────────────────────────────

_POOL_SMALL  = 20    # small pool → high cache hit rate (for CachedUser)
_POOL_LARGE  = 5_000 # large pool → low cache hit rate (for BaselineUser)

def _rand_hotel(seed: int | None = None) -> dict:
    """Random hotel feature dict.  If seed is given, output is deterministic."""
    rng = random.Random(seed)
    return {
        "price":          round(rng.uniform(50, 800), 2),
        "rating":         round(rng.uniform(1.0, 5.0), 1),
        "review_count":   rng.randint(1, 5_000),
        "amenity_count":  rng.randint(0, 20),
        "location_score": round(rng.uniform(0.0, 1.0), 3),
    }

def _rand_hotels(n: int) -> list[dict]:
    return [_rand_hotel() for _ in range(n)]


# ── User classes ──────────────────────────────────────────────────────────────

class BaselineUser(HttpUser):
    """
    Calls /predict with fully random inputs (no cache benefit).

    Use this to establish the baseline: raw inference latency under load.
    Ramp users from 10 → 500 and watch p99 diverge from p50.
    """
    wait_time = between(0.005, 0.05)   # 5–50 ms think time

    @task
    def predict_single(self) -> None:
        self.client.post(
            "/predict",
            json={"hotel": _rand_hotel()},
            name="/predict",
        )


class CachedUser(HttpUser):
    """
    Calls /predict/cached using a small pool of distinct hotels so the cache
    warms up quickly.

    Observe: after ~_POOL_SMALL requests, hit_rate → 1 and p99 collapses to
    sub-millisecond.  This demonstrates that caching is the most powerful
    latency trick when input cardinality is manageable.
    """
    wait_time = between(0.005, 0.05)

    # Pre-generate a small hotel pool (same across all user instances)
    _hotel_pool = [_rand_hotel(seed=i) for i in range(_POOL_SMALL)]

    @task
    def predict_cached(self) -> None:
        hotel = random.choice(self._hotel_pool)
        self.client.post(
            "/predict/cached",
            json={"hotel": hotel},
            name="/predict/cached",
        )


class ExplicitBatchUser(HttpUser):
    """
    Calls /predict/batch with a random batch size (5–25 hotels).

    Each request is heavier (more hotels) but uses fewer round trips.
    Useful for: offline scoring pipelines, search-results ranking pages.

    Watch: latency per request grows ~linearly with batch size, but
    throughput (hotels/second) grows much faster — that's the batching gain.
    """
    wait_time = between(0.01, 0.1)

    @task
    def predict_batch(self) -> None:
        n = random.randint(5, 25)
        self.client.post(
            "/predict/batch",
            json={"hotels": _rand_hotels(n)},
            name="/predict/batch",
        )


class MicroBatchedUser(HttpUser):
    """
    Calls /predict/batched with minimal think time to maximise concurrency.

    Each request looks single to this user, but the server groups concurrent
    requests into batches.  Increase users aggressively (200, 500, 1000) and
    watch /metrics:
      • avg_batch_size climbs → server is batching effectively
      • throughput grows → batching provides leverage
      • p99 stays bounded ≈ max_wait_ms (10 ms) + model latency

    This is the key demonstration: server-side batching lets you serve
    2 000 RPS with a model that only runs ~100 times/second.
    """
    wait_time = between(0.001, 0.01)   # very little think time → high concurrency

    @task
    def predict_batched(self) -> None:
        self.client.post(
            "/predict/batched",
            json={"hotel": _rand_hotel()},
            name="/predict/batched",
        )


class MixedUser(HttpUser):
    """
    Realistic traffic mix: mostly single predictions, some batches, some cached.
    Good for end-to-end soak testing.
    """
    wait_time = between(0.01, 0.1)

    _hotel_pool = [_rand_hotel(seed=i) for i in range(50)]

    @task(6)
    def single(self) -> None:
        self.client.post("/predict",        json={"hotel": _rand_hotel()},        name="/predict")

    @task(2)
    def cached(self) -> None:
        self.client.post("/predict/cached", json={"hotel": random.choice(self._hotel_pool)}, name="/predict/cached")

    @task(1)
    def batch(self) -> None:
        self.client.post("/predict/batch",  json={"hotels": _rand_hotels(random.randint(5, 15))}, name="/predict/batch")

    @task(1)
    def check_metrics(self) -> None:
        self.client.get("/metrics", name="/metrics")
