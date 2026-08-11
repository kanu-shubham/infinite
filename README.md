# infinite

A React sandbox with two independent workspaces, switchable from the top nav:

| Area | What it is |
|------|------------|
| **Hotel Listing** | Infinite scroll, list virtualisation, debounced filtering and sorting — all client-side |
| **ML Pipeline** | A full-stack machine-learning app: React front end → FastAPI backend → scikit-learn pipeline |

## ML Pipeline

Train a scikit-learn pipeline on hotel-booking data, watch each stage execute,
inspect what the model learned, then score new bookings against the registered
model.

- **Data** — dataset shape, target balance, where the missing values are, raw rows.
- **Train** — pick a target and estimator, tune its hyperparameters, launch a
  run and watch all eight stages report live.
- **Runs** — history, held-out metrics, ROC curve or predicted-vs-actual
  scatter, confusion matrix, feature importance, cross-validation spread, the
  run log, and a form that scores a booking through the stored model.

Two problems ship with it, over the same booking table:

| Target | Task | Question |
|--------|------|----------|
| `cancellation` | classification | Will this booking be cancelled before check-in? |
| `price` | regression | What nightly rate will this booking be sold at? |

The pipeline is one `sklearn.pipeline.Pipeline` — feature engineering →
imputation, scaling and one-hot encoding → estimator — fitted as a unit and
pickled as a unit, so the serving path replays exactly the transforms that were
learned at training time.

### Running it

The backend is required for the ML Pipeline tab; the Hotel Listing tab works
without it.

```bash
# terminal 1 — API on :8000
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# terminal 2 — React on :3000
npm install
npm start
```

`src/setupProxy.js` forwards `/api/*` to the backend in development, so there
is no CORS setup for local work. To point the frontend at a backend somewhere
else, set `REACT_APP_ML_API_URL`.

### Tests

```bash
npm run backend:test   # 43 pytest tests: dataset, pipeline, API
CI=true npm run build  # frontend build + lint
```

Backend details — architecture, the full API reference, configuration — are in
[`backend/README.md`](backend/README.md).

## Layout

```
src/
  features/
    hotels/            infinite scroll + virtualisation workspace
    ml/                ML pipeline workspace
      components/      forms, run views, charts
      hooks/           polling, async resources, form state
      services/        API client
  components/common/   shared spinner and error message
backend/
  app/ml/              dataset, features, models, pipeline, training, registry
  app/api/             HTTP routes
  tests/
```
