"""Serving path: score bookings with a previously registered pipeline.

Two ways in, and the difference is where the features come from:

* `predict(run, rows)` — the caller sends the feature values. Fine for a form
  or a backfill; unrealistic for production, where the client has an id and
  nothing else.
* `predict_by_entity(run, ids)` — the caller sends `["BK-003347"]` and the
  features are fetched from the online store. This is the shape real serving
  takes, and it is why a feature store exists.

Both paths load the model through `model_cache`, never from disk.
"""

import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from . import feature_store, model_cache
from .dataset import TARGETS
from .training import prepare_prediction_frame


class ModelUnavailable(RuntimeError):
    """The run exists but has no fitted pipeline to score with."""


def _require_pipeline(run: Dict[str, Any]):
    if run["status"] != "succeeded":
        raise ModelUnavailable(f"Run {run['run_id']} finished with status '{run['status']}'.")

    pipeline = model_cache.get(run["run_id"])
    if pipeline is None:
        raise ModelUnavailable(f"No model artifact stored for run {run['run_id']}.")
    return pipeline


def _score(pipeline, target_spec: Dict[str, Any], frame: pd.DataFrame) -> List[Dict[str, Any]]:
    """Run the pipeline and shape one result dict per row."""
    predictions = pipeline.predict(frame)

    probabilities = None
    if target_spec["task"] == "classification" and hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(frame)[:, 1]

    results: List[Dict[str, Any]] = []
    for index, value in enumerate(np.asarray(predictions)):
        entry: Dict[str, Any] = {"prediction": round(float(value), 4)}

        if target_spec["task"] == "classification":
            entry["label"] = (
                target_spec["positive_label"]
                if float(value) >= 0.5
                else target_spec["negative_label"]
            )
            if probabilities is not None:
                entry["probability"] = round(float(probabilities[index]), 4)

        results.append(entry)

    return results


def predict(run: Dict[str, Any], rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Score rows whose feature values arrived in the request body."""
    pipeline = _require_pipeline(run)
    target_key = run["config"]["target"]
    frame = prepare_prediction_frame(target_key, rows)
    return _score(pipeline, TARGETS[target_key], frame)


def predict_by_entity(
    run: Dict[str, Any], entity_ids: List[str]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Score bookings by id, fetching their features from the online store.

    Returns the predictions plus a timing/provenance block. That second value
    is not decoration: without per-stage timings you cannot tell a slow model
    from a slow feature lookup, and they have completely different fixes.
    """
    pipeline = _require_pipeline(run)
    target_key = run["config"]["target"]

    started = time.perf_counter()
    results = feature_store.lookup(target_key, entity_ids)
    lookup_ms = (time.perf_counter() - started) * 1000

    frame = feature_store.to_frame(target_key, results)

    started = time.perf_counter()
    scored = _score(pipeline, TARGETS[target_key], frame)
    inference_ms = (time.perf_counter() - started) * 1000

    # Attach per-entity provenance. A caller that cannot see "this prediction
    # was made from fully imputed features" will treat a miss as a real answer.
    for entry, result in zip(scored, results):
        entry["entity_id"] = result.entity_id
        entry["features_found"] = result.found
        entry["staleness_seconds"] = result.staleness_seconds
        entry["imputed_features"] = len(result.missing_features) if result.found else None

    diagnostics = {
        "backend": feature_store.backend().name,
        "feature_lookup_ms": round(lookup_ms, 3),
        "inference_ms": round(inference_ms, 3),
        "total_ms": round(lookup_ms + inference_ms, 3),
        "entities_requested": len(entity_ids),
        "entities_found": sum(1 for r in results if r.found),
        "model_cache": model_cache.stats()["hit_rate"],
    }
    return scored, diagnostics
