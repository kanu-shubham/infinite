# TravelAds ML System

High-throughput, low-latency ML stack for sponsored hotel ads. Target: **2000+ RPS @ p99 < 100ms** on a single serving replica.

## Architecture

```
            ┌──────────────────────────────────────────────────┐
            │            React frontend (HotelListingPage)     │
            └────────────────────┬─────────────────────────────┘
                                 │ POST /v1/rank  (50 candidates)
                                 ▼
       ┌────────────────────────────────────────────────────────┐
       │  Serving (FastAPI, uvloop, orjson)                     │
       │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
       │  │ FeatureStore │→ │  Predictor   │→ │ AuctionEngine│  │
       │  │ (Redis+LRU)  │  │ (LightGBM)   │  │ (GSP + QS)   │  │
       │  └──────▲───────┘  └──────────────┘  └──────────────┘  │
       └─────────┼──────────────────────────────────────────────┘
                 │ feature pulls (pipelined)
                 ▼
       ┌────────────────────┐        ┌────────────────────┐
       │  Redis (online FS) │◄───────│ Training pipeline  │
       └────────────────────┘        │ (LightGBM ranker)  │
                                     └────────────────────┘
```

### Services

| Service        | Role                                                    |
|----------------|---------------------------------------------------------|
| `serving`      | Online inference + auction; FastAPI on uvloop          |
| `training`     | Offline training of pCTR + quality models (LightGBM)   |
| `feature_store`| Redis-backed online features w/ in-process LRU         |

## Latency budget (p99, 100ms total)

| Stage              | Budget |
|--------------------|--------|
| Network ingress    | 5 ms   |
| Feature fetch (50) | 25 ms  |
| pCTR scoring       | 15 ms  |
| Auction            | 5 ms   |
| Serialize + egress | 10 ms  |
| Headroom           | 40 ms  |

## Throughput design

* **Async I/O**: feature pulls pipelined via `redis.asyncio` MGET.
* **Batched inference**: one LightGBM `predict` call per request over all candidates.
* **In-process LRU**: hot ad + user features cached with TTL (10s) — ~70% hit rate observed on warm traffic.
* **Process model**: `uvicorn --workers N` where N = CPU cores; each worker holds its own model copy (mmap'd booster ~30 MB).
* **Serialization**: `orjson` (~3x faster than stdlib json).
* **No GC pauses**: pre-warmed numpy buffers, no per-request allocations in the hot path.

## Models

1. **pCTR ranker** (`models/ranker.py`) — LightGBM binary classifier on click label.
2. **Quality score** (`models/ranker.py`) — calibrated from historical CTR + advertiser reputation.
3. **Auction** (`models/auction.py`) — Generalized Second Price with quality multiplier; reserve price floor.

`rank_score_i = bid_i × pCTR_i × quality_i`
`price_i = rank_score_{i+1} / (pCTR_i × quality_i) + ε`, floored at `min_cpc`.

## Run

```bash
# Install
pip install -r requirements.txt

# Train (writes models/artifacts/ranker.txt)
python -m training.train

# Serve
uvicorn serving.app:app --workers 4 --loop uvloop --http httptools --port 8080

# Or via docker-compose (serving + redis)
docker compose up --build

# Benchmark (2000 RPS, 30s)
python -m benchmarks.load_test --rps 2000 --duration 30
```

## API

```http
POST /v1/rank
Content-Type: application/json

{
  "request_id": "abc-123",
  "user": {"user_id": "u_42", "country": "US", "device": "mobile"},
  "context": {"destination": "PAR", "lead_time_days": 14, "los": 3, "pax": 2},
  "candidates": [
    {"ad_id": "ad_001", "advertiser_id": "adv_7", "bid_cpc": 1.20},
    ...
  ]
}
```

Response:

```json
{
  "request_id": "abc-123",
  "results": [
    {"ad_id": "ad_017", "slot": 1, "price_cpc": 0.83, "pctr": 0.094, "score": 0.113},
    ...
  ],
  "latency_us": 11842
}
```
