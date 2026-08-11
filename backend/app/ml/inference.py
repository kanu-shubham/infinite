"""Serving path: score new bookings with a previously registered pipeline."""

from typing import Any, Dict, List

import numpy as np

from . import registry
from .dataset import TARGETS
from .training import prepare_prediction_frame


class ModelUnavailable(RuntimeError):
    """The run exists but has no fitted pipeline to score with."""


def predict(run: Dict[str, Any], rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run the stored pipeline over `rows`, returning one result per row."""
    if run["status"] != "succeeded":
        raise ModelUnavailable(f"Run {run['run_id']} finished with status '{run['status']}'.")

    pipeline = registry.load_model(run["run_id"])
    if pipeline is None:
        raise ModelUnavailable(f"No model artifact stored for run {run['run_id']}.")

    target = TARGETS[run["config"]["target"]]
    frame = prepare_prediction_frame(run["config"]["target"], rows)
    predictions = pipeline.predict(frame)

    probabilities = None
    if target["task"] == "classification" and hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(frame)[:, 1]

    results: List[Dict[str, Any]] = []
    for index, value in enumerate(np.asarray(predictions)):
        entry: Dict[str, Any] = {"prediction": round(float(value), 4)}

        if target["task"] == "classification":
            entry["label"] = (
                target["positive_label"] if float(value) >= 0.5 else target["negative_label"]
            )
            if probabilities is not None:
                entry["probability"] = round(float(probabilities[index]), 4)

        results.append(entry)

    return results
