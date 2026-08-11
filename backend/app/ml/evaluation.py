"""Metric computation and model explanation for a fitted pipeline."""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from .features import transformed_feature_names

# How many points of a curve or scatter to ship to the browser. Enough to
# draw an honest shape, small enough to keep the payload light.
CURVE_POINTS = 60
SCATTER_POINTS = 300
TOP_FEATURES = 15


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _downsample(values: np.ndarray, count: int) -> np.ndarray:
    if len(values) <= count:
        return values
    indices = np.linspace(0, len(values) - 1, count).astype(int)
    return values[indices]


def classification_metrics(y_true, y_pred, y_score) -> Dict[str, Any]:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    metrics = {
        "primary": "roc_auc",
        "scores": {
            "accuracy": _round(accuracy_score(y_true, y_pred)),
            "precision": _round(precision_score(y_true, y_pred, zero_division=0)),
            "recall": _round(recall_score(y_true, y_pred, zero_division=0)),
            "f1": _round(f1_score(y_true, y_pred, zero_division=0)),
        },
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }

    if y_score is not None and len(np.unique(y_true)) > 1:
        metrics["scores"]["roc_auc"] = _round(roc_auc_score(y_true, y_score))
        fpr, tpr, _ = roc_curve(y_true, y_score)
        metrics["roc_curve"] = [
            {"fpr": _round(f, 4), "tpr": _round(t, 4)}
            for f, t in zip(_downsample(fpr, CURVE_POINTS), _downsample(tpr, CURVE_POINTS))
        ]
    else:
        metrics["primary"] = "f1"

    return metrics


def regression_metrics(y_true, y_pred) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred

    # Guard MAPE against zero-valued targets rather than emitting inf.
    nonzero = np.abs(y_true) > 1e-9
    mape = (
        float(np.mean(np.abs(residuals[nonzero] / y_true[nonzero])) * 100.0)
        if nonzero.any()
        else None
    )

    step = max(1, len(y_true) // SCATTER_POINTS)
    scatter = [
        {"actual": _round(a, 2), "predicted": _round(p, 2)}
        for a, p in zip(y_true[::step], y_pred[::step])
    ]

    return {
        "primary": "r2",
        "scores": {
            "r2": _round(r2_score(y_true, y_pred)),
            "mae": _round(mean_absolute_error(y_true, y_pred), 3),
            "rmse": _round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
            "mape": _round(mape, 3) if mape is not None else None,
        },
        "residual_summary": {
            "mean": _round(np.mean(residuals), 3),
            "std": _round(np.std(residuals), 3),
            "p05": _round(np.percentile(residuals, 5), 3),
            "p95": _round(np.percentile(residuals, 95), 3),
        },
        "scatter": scatter,
    }


def evaluate(task: str, y_true, y_pred, y_score=None) -> Dict[str, Any]:
    if task == "classification":
        return classification_metrics(y_true, y_pred, y_score)
    return regression_metrics(y_true, y_pred)


def _native_importances(pipeline: Pipeline) -> List[Dict[str, Any]] | None:
    """Read importances straight off the estimator when it exposes them."""
    model = pipeline.named_steps["model"]
    names = transformed_feature_names(pipeline.named_steps["preprocess"])
    if not names:
        return None

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        kind = "impurity"
    elif hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_, dtype=float)
        values = np.abs(coefficients.ravel() if coefficients.ndim > 1 else coefficients)
        kind = "coefficient"
    else:
        return None

    if len(values) != len(names):
        return None

    total = float(values.sum()) or 1.0
    ranked = sorted(
        ({"feature": n, "importance": _round(v / total, 5)} for n, v in zip(names, values)),
        key=lambda item: item["importance"],
        reverse=True,
    )
    for item in ranked:
        item["kind"] = kind
    return ranked[:TOP_FEATURES]


def _permutation_importances(
    pipeline: Pipeline, X, y, seed: int
) -> List[Dict[str, Any]]:
    """Fallback that works for any estimator, measured on raw input columns."""
    sample = min(len(X), 800)
    result = permutation_importance(
        pipeline,
        X.iloc[:sample],
        np.asarray(y)[:sample],
        n_repeats=3,
        random_state=seed,
        n_jobs=1,
    )
    values = np.clip(result.importances_mean, 0.0, None)
    total = float(values.sum()) or 1.0
    ranked = sorted(
        (
            {"feature": name, "importance": _round(value / total, 5), "kind": "permutation"}
            for name, value in zip(X.columns, values)
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )
    return ranked[:TOP_FEATURES]


def feature_importances(
    pipeline: Pipeline, X: pd.DataFrame, y, seed: int
) -> List[Dict[str, Any]]:
    """Best available importance ranking, cheapest source first."""
    native = _native_importances(pipeline)
    if native:
        return native
    try:
        return _permutation_importances(pipeline, X, y, seed)
    except Exception:  # pragma: no cover - importance is never worth failing a run
        return []
