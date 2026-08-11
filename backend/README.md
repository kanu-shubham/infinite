# Hotel Bookings ML Pipeline API

FastAPI service that trains and serves scikit-learn pipelines over a synthetic
hotel-bookings dataset. It backs the **ML Pipeline** tab of the React app.

## Running it

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Interactive docs are at <http://localhost:8000/docs>.

The React dev server proxies `/api/*` here (see `proxy` in the root
`package.json`), so no CORS configuration is needed for local work. When the
frontend is served from somewhere else, set `REACT_APP_ML_API_URL` on the
frontend and add that origin to `ML_CORS_ORIGINS` here.

## Tests

```bash
cd backend
python -m pytest -q
```

43 tests covering the dataset generator, the pipeline's behaviour on messy
input, hyperparameter validation, and the full HTTP surface end to end.

## The pipeline

Everything lives inside one `sklearn.pipeline.Pipeline`, which is what keeps
training and serving honest — the fitted object replays the exact transforms it
learned, so a prediction request cannot silently take a different path than the
training data did.

| Step | Implementation | What it does |
|------|----------------|--------------|
| `engineer` | `BookingFeatureEngineer` | Derives `total_nights`, `total_guests`, `weekend_ratio`, `lead_time_bucket`, `has_prior_cancellation` |
| `preprocess` | `ColumnTransformer` | Numeric: median impute → standard scale. Categorical: most-frequent impute → one-hot (`handle_unknown="ignore"`) |
| `model` | from the catalog | The supervised estimator |

A training run walks eight named stages — **ingest → validate → split →
preprocess → train → evaluate → cross-validate → register** — recording status,
duration and a note for each one. That per-stage record is what the frontend
renders as a live pipeline diagram, and it is also how failures get attributed:
a run that dies in `preprocess` is a different problem from one that dies in
`train`.

Runs execute on a background thread and are polled by the client. Fitted
pipelines are pickled to `backend/artifacts/runs/<run_id>/model.joblib` with
their metadata alongside, so history survives a restart.

## Targets

| Key | Task | Column | Question |
|-----|------|--------|----------|
| `cancellation` | classification | `is_canceled` | Will this booking be cancelled before check-in? |
| `price` | regression | `adr` | What nightly rate will this booking be sold at? |

Four estimators are available per task (linear baseline, random forest,
gradient boosting, histogram gradient boosting). The catalog in
`app/ml/models.py` declares which hyperparameters each one exposes, including
their ranges — the API validates and clamps against that declaration, and the
frontend generates its form from it, so the model zoo is defined exactly once.

## Dataset

`app/ml/dataset.py` generates the booking table deterministically from a seed.
It is synthetic on purpose, and generated with the messiness a pipeline has to
survive: missing values in three columns, a right-skewed lead time, rare
categorical levels, and a ~38% cancellation rate in line with public booking
extracts.

Both targets are produced from an explicit latent model, so the "right" answer
is known: cancellation is a logistic function of lead time, deposit type,
market segment, prior cancellations and special requests, and the nightly rate
is a linear function of hotel type, room type, meal plan, season and party
size. A trained model's feature importances should recover those drivers —
which makes the importance chart a genuine sanity check rather than decoration.

## API

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/health` | Liveness plus the stored-run count |
| `GET` | `/api/catalog` | Targets, estimators, hyperparameter specs, stage definitions |
| `GET` | `/api/dataset?target=` | Profile, feature specs, preview rows |
| `GET` | `/api/dataset/sample?target=&count=` | Random raw rows (prefills the prediction form) |
| `POST` | `/api/runs` | Queue a training run → `202` with a run id |
| `GET` | `/api/runs?status=&limit=` | Run summaries, newest first |
| `GET` | `/api/runs/{run_id}` | Full detail: stages, metrics, importances, logs |
| `DELETE` | `/api/runs/{run_id}` | Delete a run and its artifact |
| `POST` | `/api/runs/{run_id}/predict` | Score up to 100 bookings with that run's model |

### Example

```bash
curl -X POST localhost:8000/api/runs \
  -H 'Content-Type: application/json' \
  -d '{"target":"cancellation","model":"gradient_boosting_classifier","cv_folds":3}'

curl localhost:8000/api/runs/<run_id>

curl -X POST localhost:8000/api/runs/<run_id>/predict \
  -H 'Content-Type: application/json' \
  -d '{"rows":[{"lead_time":210,"deposit_type":"Non Refund","special_requests":0}]}'
```

Omitted features are imputed by the fitted pipeline, so a partial booking is a
valid request.

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `ML_ARTIFACTS_DIR` | `backend/artifacts` | Where runs and models are written |
| `ML_DATASET_ROWS` | `6000` | Default generated row count |
| `ML_DATASET_SEED` | `42` | Default generator seed |
| `ML_CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Comma-separated allowed origins |
| `ML_MAX_STORED_RUNS` | `50` | Oldest finished runs are evicted past this |
