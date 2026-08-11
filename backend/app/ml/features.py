"""Feature engineering and preprocessing stages of the pipeline.

`BookingFeatureEngineer` is a module-level class rather than a lambda or an
inline `FunctionTransformer` on purpose: the fitted pipeline is pickled to
disk by the registry, and joblib needs an importable path to rebuild it.
"""

from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Derived columns added by BookingFeatureEngineer, in the order it emits them.
DERIVED_NUMERIC_FEATURES = [
    "total_nights",
    "total_guests",
    "weekend_ratio",
    "lead_time_bucket",
    "has_prior_cancellation",
]


def _one_hot_encoder() -> OneHotEncoder:
    """OneHotEncoder with a dense output, across sklearn's API rename."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


class BookingFeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds booking-domain features that the raw columns only imply.

    Stateless with respect to the training data — it derives everything
    row-wise — but it still lives inside the pipeline so the same columns
    appear at train time and at inference time.
    """

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803 - sklearn signature
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        frame = X.copy()

        weekend = frame.get("stays_in_weekend_nights", pd.Series(0, index=frame.index))
        week = frame.get("stays_in_week_nights", pd.Series(0, index=frame.index))
        total_nights = (weekend.fillna(0) + week.fillna(0)).astype(float)
        frame["total_nights"] = total_nights

        adults = frame.get("adults", pd.Series(0, index=frame.index)).fillna(0)
        children = frame.get("children", pd.Series(0, index=frame.index)).fillna(0)
        frame["total_guests"] = (adults + children).astype(float)

        # Guard the division: same-day bookings legitimately have zero nights.
        frame["weekend_ratio"] = np.where(
            total_nights > 0, weekend.fillna(0) / total_nights.replace(0, np.nan), 0.0
        )
        frame["weekend_ratio"] = frame["weekend_ratio"].fillna(0.0)

        lead_time = frame.get("lead_time", pd.Series(0, index=frame.index)).fillna(0)
        frame["lead_time_bucket"] = pd.cut(
            lead_time,
            bins=[-1, 7, 30, 90, 180, np.inf],
            labels=[0, 1, 2, 3, 4],
        ).astype(float)

        prior = frame.get("previous_cancellations", pd.Series(0, index=frame.index))
        frame["has_prior_cancellation"] = (prior.fillna(0) > 0).astype(float)

        return frame

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        base = list(input_features) if input_features is not None else list(
            getattr(self, "feature_names_in_", [])
        )
        return np.asarray(base + DERIVED_NUMERIC_FEATURES, dtype=object)


def build_preprocessor(
    numeric_features: List[str], categorical_features: List[str]
) -> ColumnTransformer:
    """Impute + scale numerics, impute + one-hot encode categoricals.

    `handle_unknown="ignore"` matters at serving time: a booking can arrive
    with a room type the training extract never saw, and the request should
    score rather than 500.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", _one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features + DERIVED_NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def transformed_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Column names after encoding, used to label feature importances."""
    try:
        return [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception:  # pragma: no cover - very old sklearn fallback
        return []
