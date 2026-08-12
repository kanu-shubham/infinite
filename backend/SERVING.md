# The serving path — feature store and inference latency

Two changes turn `/predict` from a demo into something production-shaped:

1. **An online feature store.** The caller sends `{"entity_ids": ["BK-003347"]}`,
   not 20 feature values. Features are fetched server-side, in one round trip.
2. **A warm model cache.** The fitted pipeline is loaded once and held in
   memory, not unpickled from disk on every request.

Both are measured below, on this machine, against real Redis.

## The two endpoints

```bash
# 1 · Load features into the online store (a scheduled batch job in production)
curl -X POST localhost:8000/api/feature-store/materialize \
  -H 'Content-Type: application/json' -d '{"target":"cancellation"}'
# → {"backend":"redis","entities":6000,"features_per_entity":20,
#    "schema_hash":"28d0e74ec31cabf2","ttl_seconds":3600}

# 2 · Score by id — the production shape
curl -X POST localhost:8000/api/runs/<run_id>/predict-by-id \
  -H 'Content-Type: application/json' -d '{"entity_ids":["BK-000001","BK-999999"]}'
```

```json
{
  "predictions": [
    {"prediction": 1.0, "label": "Cancelled", "probability": 0.5513,
     "entity_id": "BK-000001", "features_found": true,
     "staleness_seconds": 0.39, "imputed_features": 0},
    {"prediction": 0.0, "label": "Honoured", "probability": 0.2414,
     "entity_id": "BK-999999", "features_found": false,
     "staleness_seconds": null, "imputed_features": null}
  ],
  "diagnostics": {"backend": "redis", "feature_lookup_ms": 0.627,
                  "inference_ms": 30.226, "entities_found": 2}
}
```

The original `/predict` still exists. It is the right shape for a form or a
backfill; it is the wrong shape for a service, because a real caller has an id.

## The four questions

A feature store is a cache with a nicer name unless it answers these.

### 1 · How do features get in?

`materialize()` reads through **`feature_columns(target)`** — the same function
`training.py:172` uses to build `X`. The write path and the training path cannot
select different columns, because they call the same code.

In production this is a scheduled job (the Airflow DAG next door would own it),
reading from the warehouse rather than a generator.

### 2 · How stale are they?

Two independent mechanisms, answering two different questions:

| | Question | Mechanism |
|---|---|---|
| **TTL** | Is it gone? | Redis expiry, 1h default. Expired → miss |
| **`staleness_seconds`** | How old is it? | `materialized_at` on every row, reported on every response |

`FEATURE_MAX_STALENESS_SECONDS` adds a hard refusal above an age even if the TTL
has not fired — useful when serving a stale feature is worse than not serving.

### 3 · What happens on a miss?

Nothing silently. Three properties, all tested:

- The response still contains **one prediction per requested id**, in order. A
  dropped row would misalign answers with the questions that asked for them.
- The miss is flagged: `features_found: false`.
- The prediction is still produced, from an all-`NaN` row that the pipeline's
  own imputers fill — the same medians learned at training time.

That last point is a judgement call, not an obvious default. Serving a fully
imputed prediction is right for ad ranking (a mediocre score beats no ad) and
wrong for a credit decision. The flag is what lets the caller decide.

### 4 · How does it stay consistent with training?

**This is the one that matters.** Every row is written with a fingerprint:

```python
def schema_hash(feature_names):
    return sha256("|".join(sorted(feature_names))).hexdigest()[:16]
```

Every read recomputes the hash from what the *model* expects and compares. A
mismatch raises `FeatureStoreError` → **HTTP 503**.

Without it, adding a feature to the model while the store still holds the old
set produces no error at all. Columns silently misalign, and the model gets
quietly worse. That is **online/offline skew**, and it is the most expensive
class of bug in production ML precisely because nothing crashes.

503 rather than 500 is deliberate: the model is healthy, the feature layer is
not serving safely. A caller can retry or fall back.

---

## Where the milliseconds go

`python -m benchmarks.serving_latency`. Real Redis, gradient boosting, 20
features → 59 columns.

### Model loading

| | p50 | p99 |
|---|---:|---:|
| `joblib.load` per request | 5.113 ms | 7.104 ms |
| Warm cache | 0.000 ms | 0.002 ms |

A 208 KB artifact costs 5ms to unpickle. On a 25ms budget that is 20% of the
request, spent re-reading something that never changed.

### Feature lookup — batch, always

| entities | p50 | per entity |
|---:|---:|---:|
| 1 | 0.190 ms | 0.190 ms |
| 10 | 0.341 ms | 0.034 ms |
| 100 | 1.511 ms | 0.015 ms |
| 1000 | 13.497 ms | 0.014 ms |

```
100 lookups one-by-one:  23.3 ms
100 lookups batched:      2.2 ms      → 11x
```

`MGET` is one round trip whatever the batch size. Sequential `GET`s are N round
trips. This is why the lookup API takes a list.

### Inference — and the surprise

Profiling a **single row** through the pipeline:

| stage | p50 |
|---|---:|
| feature lookup | 0.215 ms |
| build the DataFrame | 2.752 ms |
| `engineer.transform` | 4.224 ms |
| `preprocess.transform` | 7.800 ms |
| **`model.predict`** | **0.510 ms** |
| **full pipeline** | **13.862 ms** |

**The model is 4% of inference time.** The other 96% is pandas and sklearn
plumbing.

And it barely moves with batch size:

| rows | full `pipeline.predict` |
|---:|---:|
| 1 | 13.9 ms |
| 100 | 15.3 ms |

100 rows cost 10% more than 1. It is almost entirely **fixed per-call overhead** —
DataFrame construction, column selection, dtype checks — not per-row work.

Two conclusions follow, and they are the whole answer to "how do I make
inference fast":

**Batch aggressively.** If 100 rows cost the same as 1, then micro-batching
under load is nearly free throughput. This is what real serving stacks do:
collect requests for 5ms, score them together.

**Get pandas off the serving path.** Which leads to:

### ONNX — 169x

Exporting the preprocessing + estimator to ONNX and running it under
`onnxruntime`:

| batch | sklearn p50 | ONNX p50 | speedup |
|---:|---:|---:|---:|
| 1 | 8.602 ms | **0.051 ms** | **169x** |
| 10 | 8.746 ms | 0.092 ms | 95x |
| 100 | 9.623 ms | 0.499 ms | 19x |

```
max |sklearn − onnx| probability difference: 1.8e-07
```

Same answers, 169x faster on the single-row path that matters for real-time
serving. ONNX is a static computation graph: no pandas, no Python object
allocation, no dtype negotiation per call.

**Two real obstacles hit while doing this**, both worth knowing before you plan
an ONNX migration:

1. `SimpleImputer(strategy="most_frequent")` on string columns **does not
   convert** — `ValueError: could not convert string to float: 'City Hotel'`.
   The fix is to move categorical imputation upstream of the exported graph, so
   the served portion contains only convertible operators.
2. `BookingFeatureEngineer` is custom pandas and is **invisible to ONNX**. It has
   to be reimplemented — in the caller, or as ONNX operators, or in whatever
   language the serving tier speaks.

Which is the real lesson: **exportability is a design constraint on the
pipeline**, not a conversion step you bolt on afterwards. A pipeline written
without it in mind will not convert, and you find out late.

---

## What is still missing

Honest list. None of this is built here.

| Gap | Why it matters |
|---|---|
| **Redis is a single point of failure** | No Sentinel/Cluster, no read replicas. Redis down = every prediction is a total miss |
| **No circuit breaker** | A slow Redis blocks the request. Needs a timeout + fallback (the client has a 250ms socket timeout; the request has no budget of its own) |
| **No prediction logging** | Nothing records what was predicted, so live accuracy can never be measured |
| **Point-in-time correctness** | The store holds *current* values. Building a training set needs values *as of* the event — otherwise the future leaks into training |
| **No streaming updates** | `materialize` is batch-only. Genuinely fresh features (spend in the last minute) need Flink/Kafka |
| **One entity type** | Real systems join user + item + context features in one request |

Startup warm-up **is** handled: `main.py`'s lifespan preloads the most recent
succeeded runs, so the first request after a deploy does not pay the cold cost.
