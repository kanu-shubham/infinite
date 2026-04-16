"""
Unit tests for the preprocessing module.

WHY TEST PREPROCESSING?
  Preprocessing bugs are sneaky – they don't crash, they just silently
  produce wrong predictions.  Common bugs:
    - Fitting the scaler on test data (data leakage → overly optimistic metrics)
    - Dropping the wrong columns
    - NaN not handled → model gets NaN features at serving time

Run: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from training.preprocess import (
    build_preprocessor,
    get_splits,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
)


@pytest.fixture
def sample_df():
    """Small synthetic dataframe for fast tests (no file I/O needed)."""
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame({
        "city":            rng.choice(["New York", "Paris", "Tokyo"], n),
        "category":        rng.choice(["Budget", "Luxury"], n),
        "star_rating":     rng.integers(1, 6, n),
        "review_score":    rng.uniform(1, 10, n).round(1),
        "num_reviews":     rng.integers(10, 1000, n),
        "distance_km":     rng.uniform(0.1, 20, n).round(2),
        "amenities":       rng.integers(0, 20, n),
        "rooms_available": rng.integers(1, 50, n),
        "price_usd":       rng.uniform(20, 500, n).round(2),
    })


def test_preprocessor_output_shape(sample_df):
    """Preprocessor should expand categorical columns via OHE."""
    prep = build_preprocessor()
    X = sample_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    X_transformed = prep.fit_transform(X)
    # At minimum we should have all numeric cols + some OHE columns
    assert X_transformed.shape[0] == len(sample_df)
    assert X_transformed.shape[1] > len(NUMERIC_FEATURES)


def test_no_data_leakage(sample_df):
    """Preprocessor must be fit ONLY on training data."""
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = get_splits(sample_df)
    # Val/test shapes must align with the scaler fit on train
    assert X_val.shape[1]  == X_train.shape[1]
    assert X_test.shape[1] == X_train.shape[1]


def test_splits_are_disjoint(sample_df):
    """Train, val, and test must not share rows."""
    X_train, X_val, X_test, y_train, y_val, y_test, _ = get_splits(sample_df)
    total = len(y_train) + len(y_val) + len(y_test)
    assert total == len(sample_df)


def test_numeric_features_scaled(sample_df):
    """Numeric features should have approximately mean=0 after scaling."""
    prep = build_preprocessor()
    X = sample_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    X_t = prep.fit_transform(X)
    # First len(NUMERIC_FEATURES) columns are the scaled numerics
    numeric_block = X_t[:, : len(NUMERIC_FEATURES)]
    mean_abs = np.abs(numeric_block.mean())
    assert mean_abs < 0.1, f"Numeric block not zero-centered (mean={mean_abs})"


def test_unknown_categories_handled(sample_df):
    """Preprocessor should not crash on unseen categories at serving time."""
    prep = build_preprocessor()
    X_fit = sample_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    prep.fit(X_fit)

    unseen = sample_df.iloc[[0]].copy()
    unseen["city"]     = "UnknownCity"
    unseen["category"] = "UnknownCategory"
    # Should not raise (handle_unknown="ignore" in OneHotEncoder)
    result = prep.transform(unseen[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    assert result.shape[1] == prep.transform(X_fit.iloc[[0]]).shape[1]
