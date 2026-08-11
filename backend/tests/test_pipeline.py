import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from app.ml.dataset import feature_columns, generate_bookings
from app.ml.features import DERIVED_NUMERIC_FEATURES, BookingFeatureEngineer
from app.ml.models import build_estimator, resolve_hyperparameters
from app.ml.pipeline import build_pipeline
from app.ml.training import prepare_prediction_frame


@pytest.fixture(scope="module")
def bookings():
    return generate_bookings(800, seed=5)


def _fit_pipeline(bookings, model_key="logistic_regression", target="cancellation"):
    columns = feature_columns(target)
    features = columns["numeric"] + columns["categorical"]
    target_column = "is_canceled" if target == "cancellation" else "adr"
    pipeline = build_pipeline(
        columns["numeric"],
        columns["categorical"],
        model_key,
        resolve_hyperparameters(model_key, {}),
        seed=0,
    )
    pipeline.fit(bookings[features], bookings[target_column])
    return pipeline, bookings[features]


def test_feature_engineer_adds_every_derived_column(bookings):
    engineered = BookingFeatureEngineer().fit_transform(bookings)
    for column in DERIVED_NUMERIC_FEATURES:
        assert column in engineered.columns
    assert engineered["total_nights"].notna().all()


def test_feature_engineer_survives_zero_night_bookings():
    frame = pd.DataFrame(
        {
            "stays_in_weekend_nights": [0],
            "stays_in_week_nights": [0],
            "adults": [2],
            "children": [np.nan],
            "lead_time": [0],
            "previous_cancellations": [0],
        }
    )
    engineered = BookingFeatureEngineer().fit_transform(frame)
    assert engineered["weekend_ratio"].iloc[0] == 0.0
    assert engineered["total_guests"].iloc[0] == 2.0


def test_feature_engineer_does_not_mutate_its_input(bookings):
    before = bookings.copy()
    BookingFeatureEngineer().fit_transform(bookings)
    pd.testing.assert_frame_equal(bookings, before)


def test_pipeline_imputes_missing_values_rather_than_failing(bookings):
    pipeline, features = _fit_pipeline(bookings)
    with_holes = features.head(5).copy()
    with_holes.loc[:, "children"] = np.nan
    with_holes.loc[:, "region"] = None
    assert len(pipeline.predict(with_holes)) == 5


def test_pipeline_tolerates_unseen_categorical_levels(bookings):
    pipeline, features = _fit_pipeline(bookings)
    novel = features.head(3).copy()
    novel.loc[:, "room_type"] = "ZZ-penthouse"
    assert len(pipeline.predict(novel)) == 3


def test_classification_pipeline_beats_the_majority_class(bookings):
    pipeline, features = _fit_pipeline(bookings)
    predictions = pipeline.predict(features)
    assert set(np.unique(predictions)).issubset({0, 1})
    assert (predictions == bookings["is_canceled"]).mean() > 0.6


def test_regression_pipeline_learns_the_rate_signal(bookings):
    pipeline, features = _fit_pipeline(bookings, "ridge", "price")
    predictions = pipeline.predict(features)
    correlation = np.corrcoef(predictions, bookings["adr"])[0, 1]
    assert correlation > 0.8


def test_unfitted_pipeline_refuses_to_predict(bookings):
    columns = feature_columns("cancellation")
    pipeline = build_pipeline(
        columns["numeric"], columns["categorical"], "ridge", {"alpha": 1.0}, seed=0
    )
    with pytest.raises(NotFittedError):
        pipeline.predict(bookings.head(2))


def test_hyperparameters_are_clamped_and_unknown_keys_dropped():
    resolved = resolve_hyperparameters(
        "random_forest_classifier",
        {"n_estimators": 99999, "max_depth": 0, "bogus": "drop me"},
    )
    assert resolved["n_estimators"] == 600
    assert resolved["max_depth"] is None  # 0 means "unlimited"
    assert "bogus" not in resolved


def test_non_numeric_hyperparameter_falls_back_to_the_default():
    resolved = resolve_hyperparameters("ridge", {"alpha": "not-a-number"})
    assert resolved["alpha"] == 1.0


def test_class_weight_none_is_passed_through_as_python_none():
    estimator = build_estimator(
        "logistic_regression",
        resolve_hyperparameters("logistic_regression", {"class_weight": "none"}),
        seed=1,
    )
    assert estimator.class_weight is None


def test_prediction_frame_fills_absent_columns_with_nan():
    frame = prepare_prediction_frame("cancellation", [{"lead_time": 30}])
    columns = feature_columns("cancellation")
    assert list(frame.columns) == columns["numeric"] + columns["categorical"]
    assert frame["lead_time"].iloc[0] == 30
    assert frame["adr"].isna().all()


def test_prediction_frame_coerces_numeric_strings():
    frame = prepare_prediction_frame("cancellation", [{"lead_time": "45", "adults": "2"}])
    assert frame["lead_time"].iloc[0] == 45.0
    assert frame["adults"].iloc[0] == 2.0
