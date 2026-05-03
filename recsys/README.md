# Video Recommender — model + service + API

Backend-only video recommendation system. No UI in this folder; the UI is
expected to call the HTTP API defined in `api/`.

## Layout

```
recsys/
├── data/
│   └── catalog.js                # video catalogue + tag vocab + generator
├── model/                        # pure algorithmic layer (no I/O)
│   ├── similarity.js             # cosine, decay
│   ├── userProfile.js            # immutable profile + applyEvent
│   ├── candidateGenerators.js    # heuristic + learned (auto-routed)
│   ├── learnedCandidates.js      # MF item-item NN + per-user retrieval
│   ├── learnedRanker.js          # inference-time wrapper around trained ranker
│   ├── learnedModel.js           # loads artifacts; serves embeddings & scores
│   ├── ranker.js                 # heuristic linear scorer (fallback)
│   └── reranker.js               # MMR diversity + hard filters
├── service/                      # state + orchestration
│   ├── profileStore.js           # pluggable profile storage (in-memory impl)
│   ├── eventLog.js               # telemetry sink with subscribe()
│   └── recommender.js            # façade the API consumes
├── training/                     # offline ML training pipeline
│   ├── dataGenerator.js          # synthetic interaction events with labels
│   ├── matrixFactorization.js    # BPR implicit-feedback CF (SGD)
│   ├── rankerTrainer.js          # logistic regression ranker (SGD + L2)
│   ├── evaluate.js               # AUC, log-loss, Recall@K, NDCG@K
│   └── runTrain.js               # orchestrator (data → MF → ranker → eval)
├── artifacts/                    # serialised trained model (committed)
│   ├── mf.json                   # user/item embeddings + item bias
│   ├── ranker.json               # logistic-regression weights + mfScore norm
│   └── meta.json                 # training metadata + offline metrics
├── api/                          # transport layer
│   ├── routes.js                 # transport-agnostic route table
│   ├── server.js                 # Node http server (auto-loads artifacts)
│   └── client.js                 # sample fetch client (what the UI imports)
└── demo/
    └── runDemo.js                # end-to-end CLI driver
```

## ML training pipeline

```
synthetic users + catalogue
        │
        ▼
  generate interactions  ──►  train/test split (80/20 per user)
        │
        ├─► train BPR matrix factorisation (user/item embeddings, item bias)
        │       triplets (u, i+, i-) drawn from positive-engagement events
        │       loss = -log σ(score(u,i+) - score(u,i-)) + λ·||θ||²
        │
        ├─► train logistic-regression ranker over engineered features
        │       features = [mfScore, tagOverlap, channelHistory, popularity, recency]
        │       loss = binary cross-entropy + L2
        │
        ├─► evaluate on held-out set
        │       pointwise: AUC, log-loss
        │       top-K:     Recall@10, NDCG@10  (vs random baseline)
        │
        └─► persist artifacts to recsys/artifacts/
```

Run: `node recsys/training/runTrain.js` — completes in ~0.4s.

Most recent run:

| metric            | value  | random baseline |
| ----------------- | ------ | --------------- |
| Pointwise AUC     | 0.643  | 0.500           |
| Recall@10         | 0.141  | 0.046           |
| NDCG@10           | 0.093  | —               |

## Online pipeline (serving)

```
client event ──► /v1/events ──► EventLog ──► Recommender._applyEventToProfile
                                              └─► ProfileStore (updated profile)

UI request ──► /v1/recommendations ──► Recommender.recommend(userId)
                  │
                  ├─ generateCandidates(profile, catalog, learnedModel?)
                  │   ├─ if learnedModel:
                  │   │     ├─ learnedCollabCandidates (item-item NN in MF space)
                  │   │     └─ learnedUserCandidates    (userEmb·itemEmb if known)
                  │   └─ else:
                  │         ├─ contentCandidates       (cosine on tag vector)
                  │         └─ collaborativeCandidates (item-item NN on tag vec)
                  │   plus always:
                  │         ├─ trendingCandidates      (cold-start + safety net)
                  │         └─ freshCandidates         (recency)
                  │
                  ├─ applyHardFilters (drop watched / skipped / blocklist)
                  ├─ rank
                  │     learned: logistic regression over engineered features
                  │              (cold users get neutralised mfScore = train mean)
                  │     fallback: hand-set linear model
                  └─ diversify        (MMR penalty on repeated channel/tag)
                              └─► paginated slate
```

The server prefers the learned model when `recsys/artifacts/` exists and
silently falls back to the heuristic ranker otherwise. Both produce the same
output shape so the UI doesn't need to care.

## API (UI contract)

| Method | Path                         | Purpose                           |
| ------ | ---------------------------- | --------------------------------- |
| GET    | `/v1/recommendations`        | Paginated feed for a user         |
| POST   | `/v1/events`                 | Log watch / like / skip event     |
| GET    | `/v1/profile`                | Inspect a user's learned profile  |
| DELETE | `/v1/profile`                | Reset profile                     |
| GET    | `/health`                    | Liveness                          |

`GET /v1/recommendations?userId=u1&page=1&pageSize=8` →

```json
{
  "items": [
    {
      "videoId": "vid_3a",
      "title": "...",
      "channelId": "ch_neuralnotes",
      "channelName": "Neural Notes",
      "tags": ["ai", "science"],
      "durationSec": 612,
      "uploadedAt": 1714000000000,
      "source": "content",
      "rankerScore": 3.14,
      "finalScore": 2.97,
      "reasons": ["affinity", "channelAffinity"]
    }
  ],
  "hasMore": true,
  "page": 1,
  "pageSize": 8,
  "debug": { "candidatePoolSize": 180, "afterFilter": 175, "slateSize": 16 }
}
```

`POST /v1/events`:

```json
{ "userId": "u1", "videoId": "vid_3a", "type": "watch", "watchRatio": 0.85 }
```

## Run

```
node recsys/api/server.js          # starts the API on :8787
node recsys/demo/runDemo.js        # runs an end-to-end simulation
```

## Swap-points (production)

- `model/candidateGenerators.js` — replace cosine retrieval with an ANN index
  (Faiss/ScaNN) over learned user/item embeddings.
- `model/ranker.js` — replace the linear scorer with a GBDT or NN ranker;
  feature extraction stays put.
- `service/profileStore.js` — replace the in-memory map with Redis / Bigtable.
- `service/eventLog.js` — replace with a Kafka producer; offline trainer
  becomes the consumer.
