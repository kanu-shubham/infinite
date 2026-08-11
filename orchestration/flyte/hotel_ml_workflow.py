"""The eight training stages, expressed as a Flyte workflow.

Run it locally, no cluster required:

    pyflyte run orchestration/flyte/hotel_ml_workflow.py training_workflow \\
        --target cancellation --model gradient_boosting_classifier --cv_folds 3

The ML itself is unchanged — every task calls straight into `backend/app/ml`.
That is the point of keeping `ml/` free of HTTP: FastAPI was one caller, and
an orchestrator is simply another.

What changes is the *boundaries*. In `training.py` the stages are function
calls in one process, so `pipeline[:-1]` can share live objects with the full
pipeline. Here each task is potentially a separate container on a separate
machine, so everything crossing a task boundary has to serialise. That cost is
visible below: the half-fitted pipeline is written to a file between
`preprocess` and `train`.
"""

import json
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional

import joblib
import pandas as pd
from flytekit import Resources, task, workflow
from flytekit.types.file import FlyteFile

BACKEND = Path(__file__).resolve().parents[2] / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.ml.dataset import TARGETS, feature_columns, load_bookings  # noqa: E402
from app.ml.evaluation import evaluate, feature_importances  # noqa: E402
from app.ml.models import resolve_hyperparameters  # noqa: E402
from app.ml.pipeline import build_pipeline  # noqa: E402

# Every task declares what it needs. In a cluster this becomes the pod request,
# which is how one heavy training step stops sizing the whole workflow.
SMALL = Resources(cpu="1", mem="1Gi")
LARGE = Resources(cpu="2", mem="4Gi")


class SplitOutput(NamedTuple):
    """Flyte needs a named type to return more than one value from a task."""

    train: pd.DataFrame
    test: pd.DataFrame


class EvalOutput(NamedTuple):
    metrics: dict
    # `list` on its own is rejected: "Type of Generic List type is not
    # supported". Flyte needs a concrete element type so it can build a schema
    # for the value crossing the task boundary. Airflow would have accepted the
    # bare annotation and failed later, at runtime, on serialisation.
    importances: List[dict]


# ── 1 · Ingest ───────────────────────────────────────────────────────────────
# `cache=True` is the feature that has no equivalent in training.py: Flyte
# hashes the inputs, and a repeat call with the same (rows, seed) skips
# execution entirely and reuses the stored output.
@task(cache=True, cache_version="1.0", requests=SMALL, retries=2)
def ingest(rows: int, seed: int) -> pd.DataFrame:
    return load_bookings(rows, seed).copy()


# ── 2 · Validate ─────────────────────────────────────────────────────────────
@task(requests=SMALL, retries=1)
def validate(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    spec = TARGETS[target]
    columns = feature_columns(target)
    expected = columns["numeric"] + columns["categorical"] + [spec["column"]]

    missing = [c for c in expected if c not in frame]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    frame = frame.dropna(subset=[spec["column"]])
    if len(frame) < 50:
        raise ValueError("Not enough labelled rows to train on.")
    if spec["task"] == "classification" and frame[spec["column"]].nunique() < 2:
        raise ValueError("Target has a single class; nothing to learn.")

    return frame


# ── 3 · Split ────────────────────────────────────────────────────────────────
@task(requests=SMALL)
def split(frame: pd.DataFrame, target: str, test_size: float, seed: int) -> SplitOutput:
    from sklearn.model_selection import train_test_split

    spec = TARGETS[target]
    stratify = frame[spec["column"]] if spec["task"] == "classification" else None
    train, test = train_test_split(
        frame, test_size=test_size, random_state=seed, stratify=stratify
    )
    return SplitOutput(train=train, test=test)


# ── 4 · Preprocess ───────────────────────────────────────────────────────────
# The boundary tax. In-process, `pipeline[:-1]` shares step instances with the
# full pipeline, so fitting the prefix leaves the whole object half-fitted for
# free. Across a task boundary there is no shared memory, so the partially
# fitted pipeline is pickled and handed to the next task as a file.
@task(requests=LARGE, retries=1)
def preprocess(
    train: pd.DataFrame,
    target: str,
    model: str,
    hyperparameters: Optional[dict],
    seed: int,
) -> FlyteFile:
    columns = feature_columns(target)
    features = columns["numeric"] + columns["categorical"]

    pipeline = build_pipeline(
        columns["numeric"],
        columns["categorical"],
        model,
        resolve_hyperparameters(model, hyperparameters or {}),
        seed,
    )
    pipeline[:-1].fit_transform(train[features], train[TARGETS[target]["column"]])

    path = "/tmp/flyte-preprocessed.joblib"
    joblib.dump(pipeline, path)
    return FlyteFile(path)


# ── 5 · Train ────────────────────────────────────────────────────────────────
@task(requests=LARGE, retries=1)
def train_model(
    preprocessed: FlyteFile, train: pd.DataFrame, target: str
) -> FlyteFile:
    pipeline = joblib.load(preprocessed.download())
    columns = feature_columns(target)
    features = columns["numeric"] + columns["categorical"]

    matrix = pipeline[:-1].transform(train[features])
    pipeline.named_steps["model"].fit(matrix, train[TARGETS[target]["column"]])

    path = "/tmp/flyte-model.joblib"
    joblib.dump(pipeline, path)
    return FlyteFile(path)


# ── 6 · Evaluate ─────────────────────────────────────────────────────────────
@task(requests=SMALL)
def evaluate_model(
    model_file: FlyteFile, test: pd.DataFrame, train: pd.DataFrame, target: str, seed: int
) -> EvalOutput:
    pipeline = joblib.load(model_file.download())
    spec = TARGETS[target]
    columns = feature_columns(target)
    features = columns["numeric"] + columns["categorical"]

    X_test, y_test = test[features], test[spec["column"]]
    y_pred = pipeline.predict(X_test)

    y_score = None
    if spec["task"] == "classification" and hasattr(pipeline, "predict_proba"):
        y_score = pipeline.predict_proba(X_test)[:, 1]

    metrics = evaluate(spec["task"], y_test, y_pred, y_score)
    importances = feature_importances(
        pipeline, train[features], train[spec["column"]], seed
    )
    return EvalOutput(metrics=metrics, importances=importances)


# ── 7 · Cross-validate ───────────────────────────────────────────────────────
@task(requests=LARGE)
def cross_validate(
    model_file: FlyteFile, train: pd.DataFrame, target: str, folds: int
) -> dict:
    if folds < 2:
        return {"skipped": True, "reason": "Disabled for this run"}

    import numpy as np
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score

    pipeline = joblib.load(model_file.download())
    spec = TARGETS[target]
    columns = feature_columns(target)
    features = columns["numeric"] + columns["categorical"]
    scoring = "roc_auc" if spec["task"] == "classification" else "r2"

    scores = cross_val_score(
        clone(pipeline), train[features], train[spec["column"]], cv=folds, scoring=scoring
    )
    return {
        "skipped": False,
        "folds": folds,
        "scoring": scoring,
        "scores": [round(float(s), 4) for s in scores],
        "mean": round(float(np.mean(scores)), 4),
        "std": round(float(np.std(scores)), 4),
    }


# ── 8 · Register ─────────────────────────────────────────────────────────────
@task(requests=SMALL, retries=2)
def register(
    model_file: FlyteFile, metrics: dict, cross_validation: dict, target: str, model: str
) -> str:
    from app.ml import registry

    run_id = registry.next_run_id()
    pipeline = joblib.load(model_file.download())
    registry.save_model(run_id, pipeline)

    registry.create(
        {
            "run_id": run_id,
            "name": f"{model} · {TARGETS[target]['label']} (flyte)",
            "status": "succeeded",
            "created_at": registry.utcnow(),
            "finished_at": registry.utcnow(),
            "duration_ms": None,
            "progress": 1.0,
            "config": {"target": target, "task": TARGETS[target]["task"], "model": model,
                       "model_label": model, "orchestrator": "flyte"},
            "task": TARGETS[target]["task"],
            "stages": [],
            "logs": [],
            "metrics": metrics,
            "cross_validation": None if cross_validation.get("skipped") else cross_validation,
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


# ── The DAG ──────────────────────────────────────────────────────────────────
# This function body is NOT ordinary Python at execution time. Flyte runs it
# once to *compile* a graph: calling a task returns a promise, not a value, and
# the dependency edges come from which promise feeds which task. That is why
# there is no `if` on `cv_folds` here — a Python branch cannot be compiled into
# a static graph, so the skip decision lives inside the task instead.
@workflow
def training_workflow(
    target: str = "cancellation",
    model: str = "gradient_boosting_classifier",
    test_size: float = 0.2,
    cv_folds: int = 0,
    dataset_rows: int = 6000,
    dataset_seed: int = 42,
    seed: int = 42,
) -> str:
    raw = ingest(rows=dataset_rows, seed=dataset_seed)
    clean = validate(frame=raw, target=target)
    parts = split(frame=clean, target=target, test_size=test_size, seed=seed)

    preprocessed = preprocess(
        train=parts.train, target=target, model=model, hyperparameters=None, seed=seed
    )
    fitted = train_model(preprocessed=preprocessed, train=parts.train, target=target)

    scored = evaluate_model(
        model_file=fitted, test=parts.test, train=parts.train, target=target, seed=seed
    )
    cv = cross_validate(
        model_file=fitted, train=parts.train, target=target, folds=cv_folds
    )

    return register(
        model_file=fitted,
        metrics=scored.metrics,
        cross_validation=cv,
        target=target,
        model=model,
    )


if __name__ == "__main__":
    # Local execution without any Flyte infrastructure: calling the workflow
    # directly runs the tasks in-process.
    print(json.dumps({"run_id": training_workflow(cv_folds=3)}, indent=2))
