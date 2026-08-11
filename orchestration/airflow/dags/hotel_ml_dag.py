"""The eight training stages, expressed as an Airflow DAG.

    export AIRFLOW_HOME=$PWD/.airflow
    airflow dags list
    airflow tasks test hotel_ml_training ingest 2026-08-11

Same ML code as `backend/app/ml` and as the Flyte workflow beside this file.
The interesting differences are structural, and there are three:

1. **Data passing.** Airflow's XCom is a metadata-database row, sized for
   identifiers rather than datasets. A 6,000-row DataFrame does not belong in
   it, so every task writes a Parquet/joblib file and passes a *path*. Flyte
   passes the DataFrame itself as a typed output. This is the single biggest
   day-to-day difference between the two tools.

2. **Branching is real.** Airflow's graph is resolved per run, so `@task.branch`
   can genuinely skip the cross-validation task. The Flyte workflow compiles to
   a static graph, so the same decision has to live *inside* a task.

3. **Scheduling is first class.** `schedule=` and `catchup=` are on the DAG
   itself — Airflow is a scheduler that runs DAGs, where Flyte is a workflow
   engine that happens to have schedules.

Note that every heavy import sits *inside* a task. The scheduler re-parses this
file constantly, so a top-level `import sklearn` would tax every parse.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.sdk import dag, task

BACKEND = Path(__file__).resolve().parents[3] / "backend"
WORK = Path("/tmp/airflow-hotel-ml")

DEFAULT_ARGS = {
    "owner": "ml-platform",
    "retries": 2,
    "retry_delay": timedelta(seconds=30),
    "retry_exponential_backoff": True,
}

CONFIG = {
    "target": "cancellation",
    "model": "gradient_boosting_classifier",
    "test_size": 0.2,
    "cv_folds": 3,
    "dataset_rows": 6000,
    "dataset_seed": 42,
    "seed": 42,
}


def _backend_on_path() -> None:
    """Make `app.ml` importable from inside a task."""
    if str(BACKEND) not in sys.path:
        sys.path.insert(0, str(BACKEND))


@dag(
    dag_id="hotel_ml_training",
    description="Train and register a hotel-bookings model (8 stages)",
    start_date=datetime(2026, 1, 1),
    schedule="0 3 * * *",          # nightly retrain at 03:00
    catchup=False,                 # do not backfill every night since start_date
    max_active_runs=1,             # never train two models concurrently
    default_args=DEFAULT_ARGS,
    tags=["ml", "training", "hotel-bookings"],
)
def hotel_ml_training():
    # ── 1 · Ingest ───────────────────────────────────────────────────────────
    @task
    def ingest() -> str:
        _backend_on_path()
        from app.ml.dataset import load_bookings

        WORK.mkdir(parents=True, exist_ok=True)
        frame = load_bookings(CONFIG["dataset_rows"], CONFIG["dataset_seed"]).copy()
        path = WORK / "raw.parquet"
        frame.to_parquet(path)
        # Only the path crosses the XCom boundary, never the data.
        return str(path)

    # ── 2 · Validate ─────────────────────────────────────────────────────────
    @task
    def validate(raw_path: str) -> str:
        _backend_on_path()
        import pandas as pd
        from app.ml.dataset import TARGETS, feature_columns

        spec = TARGETS[CONFIG["target"]]
        columns = feature_columns(CONFIG["target"])
        frame = pd.read_parquet(raw_path)

        expected = columns["numeric"] + columns["categorical"] + [spec["column"]]
        missing = [c for c in expected if c not in frame]
        if missing:
            raise ValueError(f"Dataset is missing columns: {missing}")

        frame = frame.dropna(subset=[spec["column"]])
        if len(frame) < 50:
            raise ValueError("Not enough labelled rows to train on.")
        if spec["task"] == "classification" and frame[spec["column"]].nunique() < 2:
            raise ValueError("Target has a single class; nothing to learn.")

        path = WORK / "clean.parquet"
        frame.to_parquet(path)
        return str(path)

    # ── 3 · Split ────────────────────────────────────────────────────────────
    @task
    def split(clean_path: str) -> dict[str, str]:
        _backend_on_path()
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from app.ml.dataset import TARGETS

        spec = TARGETS[CONFIG["target"]]
        frame = pd.read_parquet(clean_path)
        stratify = frame[spec["column"]] if spec["task"] == "classification" else None
        train, test = train_test_split(
            frame,
            test_size=CONFIG["test_size"],
            random_state=CONFIG["seed"],
            stratify=stratify,
        )

        train_path, test_path = WORK / "train.parquet", WORK / "test.parquet"
        train.to_parquet(train_path)
        test.to_parquet(test_path)
        # A small dict is fine in XCom — it is two strings.
        return {"train": str(train_path), "test": str(test_path)}

    # ── 4 · Preprocess ───────────────────────────────────────────────────────
    @task
    def preprocess(paths: dict[str, str]) -> str:
        _backend_on_path()
        import joblib
        import pandas as pd
        from app.ml.dataset import TARGETS, feature_columns
        from app.ml.models import resolve_hyperparameters
        from app.ml.pipeline import build_pipeline

        columns = feature_columns(CONFIG["target"])
        features = columns["numeric"] + columns["categorical"]
        train = pd.read_parquet(paths["train"])

        pipeline = build_pipeline(
            columns["numeric"],
            columns["categorical"],
            CONFIG["model"],
            resolve_hyperparameters(CONFIG["model"], {}),
            CONFIG["seed"],
        )
        pipeline[:-1].fit_transform(
            train[features], train[TARGETS[CONFIG["target"]]["column"]]
        )

        path = WORK / "preprocessed.joblib"
        joblib.dump(pipeline, path)
        return str(path)

    # ── 5 · Train ────────────────────────────────────────────────────────────
    @task
    def train_model(preprocessed_path: str, paths: dict[str, str]) -> str:
        _backend_on_path()
        import joblib
        import pandas as pd
        from app.ml.dataset import TARGETS, feature_columns

        columns = feature_columns(CONFIG["target"])
        features = columns["numeric"] + columns["categorical"]
        train = pd.read_parquet(paths["train"])

        pipeline = joblib.load(preprocessed_path)
        matrix = pipeline[:-1].transform(train[features])
        pipeline.named_steps["model"].fit(
            matrix, train[TARGETS[CONFIG["target"]]["column"]]
        )

        path = WORK / "model.joblib"
        joblib.dump(pipeline, path)
        return str(path)

    # ── 6 · Evaluate ─────────────────────────────────────────────────────────
    @task
    def evaluate_model(model_path: str, paths: dict[str, str]) -> dict:
        _backend_on_path()
        import joblib
        import pandas as pd
        from app.ml.dataset import TARGETS, feature_columns
        from app.ml.evaluation import evaluate

        spec = TARGETS[CONFIG["target"]]
        columns = feature_columns(CONFIG["target"])
        features = columns["numeric"] + columns["categorical"]

        pipeline = joblib.load(model_path)
        test = pd.read_parquet(paths["test"])
        X_test, y_test = test[features], test[spec["column"]]

        y_pred = pipeline.predict(X_test)
        y_score = None
        if spec["task"] == "classification" and hasattr(pipeline, "predict_proba"):
            y_score = pipeline.predict_proba(X_test)[:, 1]

        metrics = evaluate(spec["task"], y_test, y_pred, y_score)
        # Metrics are small and JSON-safe, so this one really can ride XCom.
        return {"primary": metrics["primary"], "scores": metrics["scores"]}

    # ── 7 · Cross-validate, behind a branch ──────────────────────────────────
    # Airflow resolves its graph per run, so this genuinely removes a task from
    # the run. Flyte cannot do this in the workflow body.
    @task.branch
    def should_cross_validate() -> str:
        return "cross_validate" if CONFIG["cv_folds"] >= 2 else "skip_cross_validation"

    @task
    def cross_validate(model_path: str, paths: dict[str, str]) -> dict:
        _backend_on_path()
        import joblib
        import numpy as np
        import pandas as pd
        from sklearn.base import clone
        from sklearn.model_selection import cross_val_score
        from app.ml.dataset import TARGETS, feature_columns

        spec = TARGETS[CONFIG["target"]]
        columns = feature_columns(CONFIG["target"])
        features = columns["numeric"] + columns["categorical"]
        scoring = "roc_auc" if spec["task"] == "classification" else "r2"

        pipeline = joblib.load(model_path)
        train = pd.read_parquet(paths["train"])
        scores = cross_val_score(
            clone(pipeline),
            train[features],
            train[spec["column"]],
            cv=CONFIG["cv_folds"],
            scoring=scoring,
        )
        return {
            "folds": CONFIG["cv_folds"],
            "scoring": scoring,
            "scores": [round(float(s), 4) for s in scores],
            "mean": round(float(np.mean(scores)), 4),
            "std": round(float(np.std(scores)), 4),
        }

    @task
    def skip_cross_validation() -> dict:
        return {"skipped": True}

    # ── 8 · Register ─────────────────────────────────────────────────────────
    # `trigger_rule` matters here: by default a task needs every upstream to
    # succeed, and a branch leaves one of them *skipped*. Without this the
    # register step would never run.
    @task(trigger_rule="none_failed_min_one_success")
    def register(model_path: str, metrics: dict) -> str:
        _backend_on_path()
        import joblib
        from app.ml import registry
        from app.ml.dataset import TARGETS

        run_id = registry.next_run_id()
        registry.save_model(run_id, joblib.load(model_path))
        registry.create(
            {
                "run_id": run_id,
                "name": f"{CONFIG['model']} · {TARGETS[CONFIG['target']]['label']} (airflow)",
                "status": "succeeded",
                "created_at": registry.utcnow(),
                "finished_at": registry.utcnow(),
                "duration_ms": None,
                "progress": 1.0,
                "config": {**CONFIG, "task": TARGETS[CONFIG["target"]]["task"],
                           "model_label": CONFIG["model"], "orchestrator": "airflow"},
                "task": TARGETS[CONFIG["target"]]["task"],
                "stages": [],
                "logs": [],
                "metrics": metrics,
                "cross_validation": None,
                "feature_importances": [],
                "pipeline_steps": [],
                "dataset": None,
                "headline_metric": {
                    "key": metrics["primary"],
                    "label": metrics["primary"].replace("_", " ").upper(),
                    "value": metrics["scores"].get(metrics["primary"]),
                },
                "error": None,
                "failed_stage": None,
                "has_model": True,
            }
        )
        return run_id

    # ── Wiring ───────────────────────────────────────────────────────────────
    # Dependencies come from passing one task's return value to the next.
    raw = ingest()
    clean = validate(raw)
    paths = split(clean)
    preprocessed = preprocess(paths)
    model_path = train_model(preprocessed, paths)
    metrics = evaluate_model(model_path, paths)

    branch = should_cross_validate()
    cv = cross_validate(model_path, paths)
    skipped = skip_cross_validation()
    branch >> [cv, skipped]

    # Register waits for evaluation and for whichever branch ran.
    [cv, skipped] >> register(model_path, metrics)


dag_instance = hotel_ml_training()
