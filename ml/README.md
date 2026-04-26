# Feed Ranking ML System

Personalized LinkedIn-style feed ranker. Predicts P(click | user, post, context),
optimizes Normalized Cross-Entropy, serves a ranked feed via FastAPI.

## Layout

```
ml/
  feed_ranking/         core library
    config.py           hyperparams, feature dims, vocab
    schema.py           pydantic models (User, Post, Event, RankRequest)
    store.py            in-memory store for users/posts/events
    features.py         feature extractor (numeric + categorical + cross)
    data.py             synthetic data generator + PyTorch Dataset
    losses.py           BCE + Normalized Cross-Entropy
    models.py           LogisticRegression baseline + MLP
    train.py            training loop with negative downsampling
    evaluate.py         AUC, NCE, replay evaluation
    inference.py        load checkpoint + rank candidates
  api/
    main.py             FastAPI service
  scripts/
    generate_data.py    seed store with synthetic users/posts/events
    run_training.py     train + persist artefact
    serve.py            uvicorn entrypoint
  tests/                pytest unit + API tests
  artifacts/            model checkpoints + metadata (gitignored)
```

## Quickstart

```bash
pip install -r ml/requirements.txt

# 1. Seed an in-memory dataset to disk and train a model
python -m ml.scripts.generate_data --users 500 --posts 2000 --impressions 50000
python -m ml.scripts.run_training --epochs 5 --model mlp

# 2. Serve
python -m ml.scripts.serve            # uvicorn on :8000

# 3. Hit the API
curl localhost:8000/v1/health
curl localhost:8000/v1/feed/u_42?k=10
```

## REST Endpoints

| Method | Path                          | Purpose                              |
|--------|-------------------------------|--------------------------------------|
| GET    | /v1/health                    | liveness                             |
| GET    | /v1/model/info                | currently loaded model + metrics     |
| POST   | /v1/users                     | upsert user                          |
| POST   | /v1/posts                     | upsert post                          |
| POST   | /v1/events/impression         | log shown post                       |
| POST   | /v1/events/click              | log clicked post                     |
| GET    | /v1/feed/{user_id}?k=20       | ranked feed for a user               |
| POST   | /v1/model/predict             | batch score (user, post) pairs       |
| POST   | /v1/model/train               | retrain on stored events             |

## Modeling notes

- **Loss:** Normalized Cross-Entropy makes evaluation invariant to background CTR.
- **Sampling:** train-side negatives are downsampled to a configurable ratio;
  validation/test are left at their natural distribution so reported metrics
  reflect production.
- **Splits:** temporal — train on `t < t1`, validate on `[t1, t2)`, test on `[t2, T]`.
- **Replay eval:** re-rank each test impression batch and credit a click when
  the clicked post lands in top-K.
