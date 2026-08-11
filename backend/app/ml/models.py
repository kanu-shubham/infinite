"""The estimator catalog.

Each entry declares the hyperparameters the UI is allowed to tune, so the
training form is generated from this file rather than hard-coded twice.
"""

from typing import Any, Dict, List

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge


def _param(name: str, label: str, kind: str, default, **extra) -> Dict[str, Any]:
    return {"name": name, "label": label, "kind": kind, "default": default, **extra}


MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "logistic_regression": {
        "label": "Logistic Regression",
        "task": "classification",
        "description": "Linear baseline. Fast, and its coefficients are readable.",
        "factory": LogisticRegression,
        "fixed": {"max_iter": 1000},
        "params": [
            _param("C", "Inverse regularisation (C)", "float", 1.0, minimum=0.01, maximum=100.0, step=0.01),
            _param("class_weight", "Class weight", "choice", "balanced", options=["balanced", "none"]),
        ],
    },
    "random_forest_classifier": {
        "label": "Random Forest",
        "task": "classification",
        "description": "Bagged trees. Strong default, resistant to noisy features.",
        "factory": RandomForestClassifier,
        "fixed": {"n_jobs": -1},
        "params": [
            _param("n_estimators", "Trees", "int", 200, minimum=10, maximum=600, step=10),
            _param("max_depth", "Max depth (0 = unlimited)", "int", 12, minimum=0, maximum=40, step=1),
            _param("min_samples_leaf", "Min samples per leaf", "int", 2, minimum=1, maximum=50, step=1),
            _param("class_weight", "Class weight", "choice", "balanced", options=["balanced", "none"]),
        ],
    },
    "gradient_boosting_classifier": {
        "label": "Gradient Boosting",
        "task": "classification",
        "description": "Sequential boosting. Usually the accuracy ceiling here.",
        "factory": GradientBoostingClassifier,
        "fixed": {},
        "params": [
            _param("n_estimators", "Boosting rounds", "int", 150, minimum=10, maximum=600, step=10),
            _param("learning_rate", "Learning rate", "float", 0.1, minimum=0.01, maximum=1.0, step=0.01),
            _param("max_depth", "Max depth", "int", 3, minimum=1, maximum=12, step=1),
        ],
    },
    "hist_gradient_boosting_classifier": {
        "label": "Histogram Gradient Boosting",
        "task": "classification",
        "description": "Binned boosting. Scales to far more rows than the others.",
        "factory": HistGradientBoostingClassifier,
        "fixed": {},
        "params": [
            _param("max_iter", "Boosting rounds", "int", 200, minimum=10, maximum=800, step=10),
            _param("learning_rate", "Learning rate", "float", 0.1, minimum=0.01, maximum=1.0, step=0.01),
            _param("max_leaf_nodes", "Max leaf nodes", "int", 31, minimum=4, maximum=128, step=1),
        ],
    },
    "ridge": {
        "label": "Ridge Regression",
        "task": "regression",
        "description": "L2-penalised linear baseline for the nightly-rate target.",
        "factory": Ridge,
        "fixed": {},
        "params": [
            _param("alpha", "Regularisation (alpha)", "float", 1.0, minimum=0.01, maximum=100.0, step=0.01),
        ],
    },
    "random_forest_regressor": {
        "label": "Random Forest",
        "task": "regression",
        "description": "Bagged regression trees; captures non-linear rate effects.",
        "factory": RandomForestRegressor,
        "fixed": {"n_jobs": -1},
        "params": [
            _param("n_estimators", "Trees", "int", 200, minimum=10, maximum=600, step=10),
            _param("max_depth", "Max depth (0 = unlimited)", "int", 14, minimum=0, maximum=40, step=1),
            _param("min_samples_leaf", "Min samples per leaf", "int", 2, minimum=1, maximum=50, step=1),
        ],
    },
    "gradient_boosting_regressor": {
        "label": "Gradient Boosting",
        "task": "regression",
        "description": "Sequential boosting on squared error.",
        "factory": GradientBoostingRegressor,
        "fixed": {},
        "params": [
            _param("n_estimators", "Boosting rounds", "int", 200, minimum=10, maximum=600, step=10),
            _param("learning_rate", "Learning rate", "float", 0.08, minimum=0.01, maximum=1.0, step=0.01),
            _param("max_depth", "Max depth", "int", 3, minimum=1, maximum=12, step=1),
        ],
    },
    "hist_gradient_boosting_regressor": {
        "label": "Histogram Gradient Boosting",
        "task": "regression",
        "description": "Binned boosting for the regression target.",
        "factory": HistGradientBoostingRegressor,
        "fixed": {},
        "params": [
            _param("max_iter", "Boosting rounds", "int", 250, minimum=10, maximum=800, step=10),
            _param("learning_rate", "Learning rate", "float", 0.08, minimum=0.01, maximum=1.0, step=0.01),
            _param("max_leaf_nodes", "Max leaf nodes", "int", 31, minimum=4, maximum=128, step=1),
        ],
    },
}

DEFAULT_MODEL_FOR_TASK = {
    "classification": "gradient_boosting_classifier",
    "regression": "gradient_boosting_regressor",
}


def models_for_task(task: str) -> List[Dict[str, Any]]:
    """Catalog entries for one task, stripped of the non-serialisable factory."""
    return [
        {
            "key": key,
            "label": spec["label"],
            "task": spec["task"],
            "description": spec["description"],
            "params": spec["params"],
        }
        for key, spec in MODEL_CATALOG.items()
        if spec["task"] == task
    ]


def _coerce(spec: Dict[str, Any], value: Any) -> Any:
    """Cast a JSON value to what the estimator expects, clamped to its range."""
    kind = spec["kind"]

    if kind == "choice":
        if value not in spec["options"]:
            return spec["default"]
        # "none" is how the UI spells Python's None for class_weight.
        return None if value == "none" else value

    if value is None:
        return spec["default"]

    try:
        number = int(value) if kind == "int" else float(value)
    except (TypeError, ValueError):
        return spec["default"]

    minimum, maximum = spec.get("minimum"), spec.get("maximum")
    if minimum is not None:
        number = max(number, minimum)
    if maximum is not None:
        number = min(number, maximum)

    # A depth of 0 is the UI's way of saying "grow the tree out".
    if spec["name"] == "max_depth" and number == 0:
        return None

    return number


def resolve_hyperparameters(model_key: str, overrides: Dict[str, Any] | None) -> Dict[str, Any]:
    """Merge user overrides onto the catalog defaults, validating as we go.

    Unknown keys are dropped rather than forwarded — the estimator constructor
    is not a place to let arbitrary client input through.
    """
    spec = MODEL_CATALOG[model_key]
    overrides = overrides or {}
    resolved: Dict[str, Any] = {}

    for param in spec["params"]:
        name = param["name"]
        raw = overrides.get(name, param["default"])
        resolved[name] = _coerce(param, raw)

    return resolved


def build_estimator(model_key: str, hyperparameters: Dict[str, Any], seed: int):
    """Instantiate the estimator described by the catalog."""
    if model_key not in MODEL_CATALOG:
        raise KeyError(f"Unknown model '{model_key}'")

    spec = MODEL_CATALOG[model_key]
    factory = spec["factory"]
    kwargs = {**spec["fixed"], **hyperparameters}

    # Not every estimator accepts random_state (or needs it).
    if "random_state" in factory().get_params():
        kwargs["random_state"] = seed

    return factory(**kwargs)
