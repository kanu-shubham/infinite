"""Training orchestration.

`execute_run` walks the pipeline one named stage at a time and writes the
outcome of each into the run record as it goes, which is what lets the
frontend show a live pipeline diagram instead of a spinner. Stage boundaries
are also where failures get attributed: a run that dies in `preprocess` looks
different from one that dies in `train`.
"""

import time
import traceback
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, train_test_split

from . import registry
from .dataset import TARGETS, feature_columns, load_bookings
from .evaluation import evaluate, feature_importances
from .models import MODEL_CATALOG, resolve_hyperparameters
from .pipeline import build_pipeline, describe_pipeline

STAGES: List[Dict[str, str]] = [
    {"key": "ingest", "label": "Ingest", "detail": "Load the booking table"},
    {"key": "validate", "label": "Validate", "detail": "Schema and target checks"},
    {"key": "split", "label": "Split", "detail": "Hold out a test set"},
    {"key": "preprocess", "label": "Preprocess", "detail": "Fit engineering + encoders"},
    {"key": "train", "label": "Train", "detail": "Fit the estimator"},
    {"key": "evaluate", "label": "Evaluate", "detail": "Score on the held-out split"},
    {"key": "cross_validate", "label": "Cross-validate", "detail": "K-fold on the training split"},
    {"key": "register", "label": "Register", "detail": "Persist the fitted pipeline"},
]

CV_SCORING = {"classification": "roc_auc", "regression": "r2"}


class StageError(RuntimeError):
    """A stage failed; carries which one so the UI can mark it red."""

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def initial_stages() -> List[Dict[str, Any]]:
    return [
        {**stage, "status": "pending", "duration_ms": None, "note": None}
        for stage in STAGES
    ]


def build_run_record(config: Dict[str, Any], name: str | None) -> Dict[str, Any]:
    run_id = registry.next_run_id()
    target = TARGETS[config["target"]]
    model_label = MODEL_CATALOG[config["model"]]["label"]

    return {
        "run_id": run_id,
        "name": name or f"{model_label} · {target['label']}",
        "status": "queued",
        "created_at": registry.utcnow(),
        "started_at": None,
        "finished_at": None,
        "duration_ms": None,
        "progress": 0.0,
        "config": config,
        "task": target["task"],
        "stages": initial_stages(),
        "logs": [],
        "metrics": None,
        "cross_validation": None,
        "feature_importances": [],
        "pipeline_steps": [],
        "dataset": None,
        "headline_metric": None,
        "error": None,
        "failed_stage": None,
        "has_model": False,
    }


def _log(run_id: str, level: str, message: str) -> None:
    def mutate(run: Dict[str, Any]) -> None:
        run["logs"].append({"ts": registry.utcnow(), "level": level, "message": message})

    registry.update(run_id, mutate)


def _stage_start(run_id: str, key: str) -> float:
    index = next(i for i, stage in enumerate(STAGES) if stage["key"] == key)

    def mutate(run: Dict[str, Any]) -> None:
        run["stages"][index]["status"] = "running"
        run["progress"] = round(index / len(STAGES), 3)

    registry.update(run_id, mutate)
    return time.perf_counter()


def _stage_done(run_id: str, key: str, started: float, note: str | None = None,
                status: str = "succeeded") -> None:
    index = next(i for i, stage in enumerate(STAGES) if stage["key"] == key)
    elapsed = round((time.perf_counter() - started) * 1000, 1)

    def mutate(run: Dict[str, Any]) -> None:
        run["stages"][index].update(
            {"status": status, "duration_ms": elapsed, "note": note}
        )
        run["progress"] = round((index + 1) / len(STAGES), 3)

    registry.update(run_id, mutate)
    if note:
        _log(run_id, "info", f"{STAGES[index]['label']}: {note}")


def _stage_skipped(run_id: str, key: str, note: str) -> None:
    index = next(i for i, stage in enumerate(STAGES) if stage["key"] == key)

    def mutate(run: Dict[str, Any]) -> None:
        run["stages"][index].update({"status": "skipped", "note": note})
        run["progress"] = round((index + 1) / len(STAGES), 3)

    registry.update(run_id, mutate)


def _mark_failed(run_id: str, stage_key: str | None, message: str) -> None:
    def mutate(run: Dict[str, Any]) -> None:
        run["status"] = "failed"
        run["error"] = message
        run["failed_stage"] = stage_key
        run["finished_at"] = registry.utcnow()
        for stage in run["stages"]:
            if stage["key"] == stage_key:
                stage["status"] = "failed"
                stage["note"] = message
            elif stage["status"] == "pending":
                stage["status"] = "skipped"

    registry.update(run_id, mutate)
    _log(run_id, "error", message)


def execute_run(run_id: str) -> None:
    """Run the full pipeline for `run_id`. Never raises — failures land on the record."""
    run = registry.get(run_id)
    if run is None:
        return

    config = run["config"]
    target = TARGETS[config["target"]]
    task = target["task"]
    target_column = target["column"]
    seed = config["seed"]
    started_wall = time.perf_counter()

    registry.update(
        run_id,
        lambda r: r.update({"status": "running", "started_at": registry.utcnow()}),
    )
    _log(run_id, "info", f"Run started · {MODEL_CATALOG[config['model']]['label']} → {target['label']}")

    try:
        # ── Ingest ──────────────────────────────────────────────────────────
        started = _stage_start(run_id, "ingest")
        frame = load_bookings(config["dataset_rows"], config["dataset_seed"]).copy()
        _stage_done(run_id, "ingest", started, f"{len(frame):,} rows loaded")

        # ── Validate ────────────────────────────────────────────────────────
        started = _stage_start(run_id, "validate")
        columns = feature_columns(config["target"])
        feature_names = columns["numeric"] + columns["categorical"]
        missing_columns = [c for c in feature_names + [target_column] if c not in frame]
        if missing_columns:
            raise StageError("validate", f"Dataset is missing columns: {missing_columns}")

        before = len(frame)
        frame = frame.dropna(subset=[target_column])
        dropped = before - len(frame)
        if len(frame) < 50:
            raise StageError("validate", "Not enough labelled rows to train on.")
        if task == "classification" and frame[target_column].nunique() < 2:
            raise StageError("validate", "Target has a single class; nothing to learn.")
        _stage_done(
            run_id,
            "validate",
            started,
            f"{len(feature_names)} features, {dropped} row(s) dropped for a missing target",
        )

        X = frame[feature_names]
        y = frame[target_column]

        # ── Split ───────────────────────────────────────────────────────────
        started = _stage_start(run_id, "split")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config["test_size"],
            random_state=seed,
            stratify=y if task == "classification" else None,
        )
        _stage_done(
            run_id,
            "split",
            started,
            f"{len(X_train):,} train / {len(X_test):,} test ({int(config['test_size'] * 100)}% held out)",
        )

        hyperparameters = resolve_hyperparameters(config["model"], config.get("hyperparameters"))
        pipeline = build_pipeline(
            columns["numeric"], columns["categorical"], config["model"], hyperparameters, seed
        )

        # ── Preprocess ──────────────────────────────────────────────────────
        # Slicing shares the step instances, so fitting the prefix here leaves
        # the full pipeline fitted end to end once the estimator is done.
        started = _stage_start(run_id, "preprocess")
        preprocessing = pipeline[:-1]
        X_train_matrix = preprocessing.fit_transform(X_train, y_train)
        _stage_done(
            run_id,
            "preprocess",
            started,
            f"{X_train_matrix.shape[1]} columns after encoding",
        )

        # ── Train ───────────────────────────────────────────────────────────
        started = _stage_start(run_id, "train")
        estimator = pipeline.named_steps["model"]
        estimator.fit(X_train_matrix, y_train)
        _stage_done(run_id, "train", started, f"Fitted {type(estimator).__name__}")

        # ── Evaluate ────────────────────────────────────────────────────────
        started = _stage_start(run_id, "evaluate")
        y_pred = pipeline.predict(X_test)
        y_score = None
        if task == "classification" and hasattr(estimator, "predict_proba"):
            y_score = pipeline.predict_proba(X_test)[:, 1]
        metrics = evaluate(task, y_test, y_pred, y_score)
        importances = feature_importances(pipeline, X_train, y_train, seed)
        headline_key = metrics["primary"]
        headline = {
            "key": headline_key,
            "label": headline_key.replace("_", " ").upper(),
            "value": metrics["scores"].get(headline_key),
        }
        _stage_done(
            run_id,
            "evaluate",
            started,
            f"{headline['label']} = {headline['value']}",
        )

        # ── Cross-validate ──────────────────────────────────────────────────
        cross_validation = None
        folds = int(config.get("cv_folds") or 0)
        if folds >= 2:
            started = _stage_start(run_id, "cross_validate")
            scoring = CV_SCORING[task]
            scores = cross_val_score(
                clone(pipeline), X_train, y_train, cv=folds, scoring=scoring, n_jobs=1
            )
            cross_validation = {
                "folds": folds,
                "scoring": scoring,
                "scores": [round(float(s), 4) for s in scores],
                "mean": round(float(np.mean(scores)), 4),
                "std": round(float(np.std(scores)), 4),
            }
            _stage_done(
                run_id,
                "cross_validate",
                started,
                f"{scoring} = {cross_validation['mean']} ± {cross_validation['std']}",
            )
        else:
            _stage_skipped(run_id, "cross_validate", "Disabled for this run")

        # ── Register ────────────────────────────────────────────────────────
        started = _stage_start(run_id, "register")
        registry.save_model(run_id, pipeline)
        _stage_done(run_id, "register", started, "Pipeline written to the model registry")

        total_ms = round((time.perf_counter() - started_wall) * 1000, 1)
        steps = describe_pipeline(pipeline)

        def finish(record: Dict[str, Any]) -> None:
            record.update(
                {
                    "status": "succeeded",
                    "finished_at": registry.utcnow(),
                    "duration_ms": total_ms,
                    "progress": 1.0,
                    "metrics": metrics,
                    "cross_validation": cross_validation,
                    "feature_importances": importances,
                    "pipeline_steps": steps,
                    "headline_metric": headline,
                    "has_model": True,
                    "dataset": {
                        "rows": int(len(frame)),
                        "train_rows": int(len(X_train)),
                        "test_rows": int(len(X_test)),
                        "raw_features": len(feature_names),
                        "encoded_features": int(X_train_matrix.shape[1]),
                    },
                    "resolved_hyperparameters": {
                        key: ("none" if value is None else value)
                        for key, value in hyperparameters.items()
                    },
                }
            )

        registry.update(run_id, finish)
        _log(run_id, "info", f"Run finished in {total_ms:.0f} ms")

    except StageError as error:
        _mark_failed(run_id, error.stage, str(error))
    except Exception as error:  # noqa: BLE001 - the record is the error channel
        current = registry.get(run_id)
        running = next(
            (s["key"] for s in (current or {}).get("stages", []) if s["status"] == "running"),
            None,
        )
        _mark_failed(run_id, running, f"{type(error).__name__}: {error}")
        _log(run_id, "error", traceback.format_exc(limit=3))


def prepare_prediction_frame(target_key: str, rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Coerce request payloads into the exact column set the pipeline expects.

    Absent fields become NaN and are filled by the pipeline's imputers, which
    is the whole reason imputation lives inside the fitted object.
    """
    columns = feature_columns(target_key)
    expected = columns["numeric"] + columns["categorical"]
    frame = pd.DataFrame(rows)

    for column in expected:
        if column not in frame:
            frame[column] = np.nan

    frame = frame[expected]
    for column in columns["numeric"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame
