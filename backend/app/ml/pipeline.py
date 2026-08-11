"""Assembly of the end-to-end scikit-learn pipeline.

One `Pipeline` object owns feature engineering, imputation, scaling, encoding
and the estimator. Keeping all of it inside a single fitted object is what
makes the serving path safe: `predict` replays the exact transformations that
were fitted on the training split, so training and inference cannot drift.
"""

from typing import Any, Dict, List

from sklearn.pipeline import Pipeline

from .features import BookingFeatureEngineer, build_preprocessor
from .models import build_estimator

PIPELINE_STEPS = [
    {
        "key": "engineer",
        "label": "Feature engineering",
        "detail": "Derives total nights, party size, weekend ratio and lead-time buckets.",
    },
    {
        "key": "preprocess",
        "label": "Preprocessing",
        "detail": "Median/mode imputation, standard scaling, one-hot encoding.",
    },
    {
        "key": "model",
        "label": "Estimator",
        "detail": "The supervised learner fitted on the transformed matrix.",
    },
]


def build_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    model_key: str,
    hyperparameters: Dict[str, Any],
    seed: int,
) -> Pipeline:
    """Compose engineer → preprocessor → estimator into one fittable object."""
    return Pipeline(
        steps=[
            ("engineer", BookingFeatureEngineer()),
            ("preprocess", build_preprocessor(numeric_features, categorical_features)),
            ("model", build_estimator(model_key, hyperparameters, seed)),
        ]
    )


def describe_pipeline(pipeline: Pipeline) -> List[Dict[str, str]]:
    """Human-readable step list for the UI's pipeline diagram."""
    described = []
    for step in PIPELINE_STEPS:
        estimator = pipeline.named_steps.get(step["key"])
        described.append(
            {
                **step,
                "implementation": type(estimator).__name__ if estimator else "—",
            }
        )
    return described
