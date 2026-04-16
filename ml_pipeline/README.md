# Hotel Price Prediction – ML Pipeline

> A **real-world, beginner-friendly** ML pipeline demonstrating every component
> a production ML system needs: feature store, experiment tracking, pipeline
> orchestration, and CI/CD.

---

## What you'll learn

| Concept | Tool | File |
|---|---|---|
| Synthetic data generation | pandas / numpy | `data/generate_data.py` |
| Feature store | **Feast** | `feature_store/feature_repo/features.py` |
| Experiment tracking | **MLflow** | `training/train.py` |
| Model evaluation & promotion | MLflow Model Registry | `training/evaluate.py` |
| Pipeline orchestration | **Apache Airflow** | `pipeline/airflow_dag.py` |
| Model serving (REST API) | FastAPI | `serving/predict.py` |
| CI/CD for ML | GitHub Actions | `.github/workflows/` |
| Containerisation | Docker | `Dockerfile` |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      WEEKLY AIRFLOW DAG                      │
│                                                              │
│  [generate_data] → [feast_materialize] → [train_model]       │
│                                              │               │
│                                       [evaluate_model]       │
│                                        /          \          │
│                               [promote_model]  [send_alert]  │
└──────────────────────────────────────────────────────────────┘
         │                    │                   │
   Raw CSV data         Feature Store         MLflow Model
   (offline store)      (online store)         Registry
                              │                   │
                    ┌─────────▼───────────────────▼──────────┐
                    │         FastAPI Serving API             │
                    │      POST /predict  →  {"price": 145}  │
                    └────────────────────────────────────────┘
```

---

## Quick Start (5 minutes)

### 1 – Install dependencies

```bash
cd ml_pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2 – Generate data

```bash
python data/generate_data.py
# → data/hotels.csv  (2,000 synthetic hotels)
```

### 3 – Register features in Feast

```bash
cd feature_store/feature_repo
feast apply          # reads features.py, writes to registry.db
feast materialize-incremental "$(date -u +%Y-%m-%dT%H:%M:%S)"
cd ../..
```

### 4 – Train the model (logged to MLflow)

```bash
python training/train.py
# → mlruns/ directory created automatically
```

### 5 – View experiments in MLflow UI

```bash
mlflow ui
# → http://localhost:5000
```
You'll see every run with its hyperparameters, RMSE, MAE, R² – all in one table.

### 6 – Evaluate & promote model

```bash
python training/evaluate.py
# If RMSE < $35 and R² > 0.85, the model is promoted to "Production"
```

### 7 – Start the prediction API

```bash
uvicorn serving.predict:app --reload
# → http://localhost:8000/docs   (interactive Swagger UI)
```

### 8 – Make a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "hotel_id": 1,
    "city": "Paris",
    "category": "Luxury",
    "star_rating": 5,
    "review_score": 9.0,
    "num_reviews": 500,
    "distance_km": 1.0,
    "amenities": 15,
    "rooms_available": 3
  }'
# → {"hotel_id": 1, "predicted_price": 287.45, "model_version": "Production"}
```

---

## Component Deep-Dives

### Feast – Feature Store

**Problem it solves:** Without a feature store, training code and serving code
compute features independently. They drift over time (**training/serving skew**),
causing a model that looks great in training to fail silently in production.

**How Feast works:**

```
Offline store (CSV / BigQuery)          Online store (SQLite / Redis)
      │                                         │
      │   feast materialize                     │
      └────────────────────────────────────────►│
                                                │
Training:                        Serving (< 10 ms):
get_historical_features()        get_online_features()
(point-in-time correct)          (latest values only)
```

Key files:
- `feature_store/feature_repo/feature_store.yaml` – project config
- `feature_store/feature_repo/features.py` – feature definitions
- `feature_store/materialize.py` – push offline → online

### MLflow – Experiment Tracking

**Problem it solves:** "Which model is in production? What hyperparameters did
it use? Was it better than last month's model?"  Without tracking you can't
answer these questions.

**What MLflow logs automatically:**

```
mlflow.start_run()
  mlflow.log_params({"n_estimators": 300, "learning_rate": 0.05})
  mlflow.log_metrics({"test_rmse": 28.3, "test_r2": 0.91})
  mlflow.log_artifact("preprocessor.pkl")
  mlflow.xgboost.log_model(model, registered_model_name="hotel-price-predictor")
```

**MLflow Model Registry stages:**

```
None → Staging → Production → Archived
```

`evaluate.py` automatically promotes Staging → Production when quality gates pass.

### Apache Airflow – Pipeline Orchestration

**Problem it solves:** ML pipelines have many steps. If step 3 fails, you want
automatic retries, alerting, and a visual log of what ran when.

**The DAG (Directed Acyclic Graph):**

```python
start >> generate_data >> feast_materialize >> train_model \
      >> evaluate_model >> [promote_model, send_alert] >> end
```

The `>>` operator sets task dependencies. Airflow draws this as a graph in its UI.

**BranchPythonOperator:** `evaluate_model` returns either `"promote_model"` or
`"send_alert"` — Airflow only runs the matching downstream task.

### CI/CD for ML (GitHub Actions)

**CI (`ml_ci.yml`) – runs on every PR:**
1. Lint with `ruff`
2. Unit tests with `pytest` (data tests + preprocessing tests)
3. Smoke training (10 trees, finishes in seconds)
4. Feature schema validation

**CD (`ml_cd.yml`) – runs on merge to main:**
1. Full training run (300 trees)
2. Quality gate evaluation (RMSE + R² thresholds)
3. Build & push Docker image to GitHub Container Registry
4. Deploy to production (kubectl / Cloud Run)
5. Smoke test the live endpoint

**If the model fails the quality gate, `sys.exit(1)` stops the deploy.**

---

## Project Structure

```
ml_pipeline/
├── data/
│   └── generate_data.py        # Synthetic hotel dataset
├── feature_store/
│   ├── feature_repo/
│   │   ├── feature_store.yaml  # Feast project config
│   │   └── features.py         # Feature/entity definitions
│   └── materialize.py          # Offline → online sync
├── training/
│   ├── preprocess.py           # Feature engineering pipeline
│   ├── train.py                # MLflow training + logging
│   └── evaluate.py             # Quality gates + model promotion
├── pipeline/
│   └── airflow_dag.py          # Full pipeline as Airflow DAG
├── serving/
│   └── predict.py              # FastAPI prediction endpoint
├── tests/
│   ├── test_data.py
│   └── test_preprocess.py
├── Dockerfile                  # Multi-stage production container
└── requirements.txt
```

---

## Running Tests

```bash
cd ml_pipeline
pytest tests/ -v --cov=. --cov-report=term-missing
```

---

## Common Pitfalls for Beginners

| Pitfall | Fix |
|---|---|
| Fitting scaler on test set | `preprocessor.fit()` only on `X_train`, then `.transform()` on val/test |
| Training/serving skew | Use Feast — same features in training and serving |
| No experiment tracking | Every `python train.py` should call `mlflow.start_run()` |
| No quality gate | Add RMSE/R² threshold check before every deployment |
| Model never versioned | Use MLflow Model Registry; tag every production deploy |
| Single-stage Docker | Multi-stage build reduces image size 3×–5× |

---

## Next Steps

1. **Replace synthetic data** with real hotel data from a database
2. **Add feature monitoring** (Evidently AI / Whylogs) to detect data drift
3. **Add hyperparameter search** (Optuna + MLflow) to find better params automatically
4. **Add A/B testing** to compare old vs. new model in production traffic
5. **Switch to Kubernetes** for autoscaling the serving API
