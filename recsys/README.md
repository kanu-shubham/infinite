# Video Recommender — model + service + API

Backend-only video recommendation system. No UI in this folder; the UI is
expected to call the HTTP API defined in `api/`.

## Layout

```
recsys/
├── data/
│   └── catalog.js              # video catalogue + tag vocab + generator
├── model/                      # pure algorithmic layer (no I/O)
│   ├── similarity.js           # cosine, decay
│   ├── userProfile.js          # immutable profile + applyEvent
│   ├── candidateGenerators.js  # content / collab / trending / fresh
│   ├── ranker.js               # linear scoring model + feature extraction
│   └── reranker.js             # MMR diversity + hard filters
├── service/                    # state + orchestration
│   ├── profileStore.js         # pluggable profile storage (in-memory impl)
│   ├── eventLog.js             # telemetry sink with subscribe()
│   └── recommender.js          # façade the API consumes
├── api/                        # transport layer
│   ├── routes.js               # transport-agnostic route table
│   ├── server.js               # Node http server
│   └── client.js               # sample fetch client (what the UI imports)
└── demo/
    └── runDemo.js              # end-to-end CLI driver
```

## Pipeline

```
client event ──► /v1/events ──► EventLog ──► Recommender._applyEventToProfile
                                              └─► ProfileStore (updated profile)

UI request ──► /v1/recommendations ──► Recommender.recommend(userId)
                  │
                  ├─ generateCandidates(profile, catalog)
                  │     ├─ contentCandidates       (cosine on tag vector)
                  │     ├─ collaborativeCandidates (item-item NN from history)
                  │     ├─ trendingCandidates      (cold-start + safety net)
                  │     └─ freshCandidates         (recency)
                  │
                  ├─ applyHardFilters (drop watched / skipped / blocklist)
                  ├─ rank             (linear model over features)
                  └─ diversify        (MMR penalty on repeated channel/tag)
                              └─► paginated slate
```

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
