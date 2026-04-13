"""
Tests for model training and prediction quality.

The quality-gate test (R² > 0.85) mirrors the CI check in evaluate.py,
so a regression is caught both locally and in the pipeline.
"""
import numpy as np
import pytest
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.data_pipeline import FEATURE_COLUMNS, TARGET_COLUMN, build_preprocessor, generate_hotel_data


@pytest.fixture(scope="module")
def trained_pipeline():
    """Train once per test module to keep the suite fast."""
    df = generate_hotel_data(n_samples=2_000, random_state=42)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", XGBRegressor(n_estimators=50, max_depth=4, random_state=42)),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline, X_test, y_test


class TestModelQuality:
    def test_r2_above_production_threshold(self, trained_pipeline):
        pipeline, X_test, y_test = trained_pipeline
        r2 = r2_score(y_test, pipeline.predict(X_test))
        assert r2 > 0.85, (
            f"R² {r2:.4f} is below the 0.85 production threshold. "
            "Investigate feature engineering or hyperparameters."
        )

    def test_predictions_are_always_positive(self, trained_pipeline):
        pipeline, X_test, _ = trained_pipeline
        preds = pipeline.predict(X_test)
        assert (preds > 0).all(), "Model produced non-positive price predictions"

    def test_predictions_within_realistic_range(self, trained_pipeline):
        pipeline, X_test, _ = trained_pipeline
        preds = pipeline.predict(X_test)
        assert preds.max() < 5_000, "Predictions unrealistically high (> $5,000)"
        assert preds.min() > 0, "Predictions must be positive"

    def test_pipeline_predict_returns_correct_shape(self, trained_pipeline):
        pipeline, X_test, _ = trained_pipeline
        preds = pipeline.predict(X_test)
        assert preds.shape == (len(X_test),)

    def test_five_star_costs_more_than_three_star(self, trained_pipeline):
        """Sanity check: the model learned the right direction for star rating."""
        pipeline, _, _ = trained_pipeline
        df = generate_hotel_data(n_samples=200, random_state=0)
        base = df.iloc[:10].copy()

        three_star = base.copy()
        three_star["star_rating"] = 3
        five_star = base.copy()
        five_star["star_rating"] = 5

        pred_3 = pipeline.predict(three_star[FEATURE_COLUMNS]).mean()
        pred_5 = pipeline.predict(five_star[FEATURE_COLUMNS]).mean()
        assert pred_5 > pred_3, "5-star hotels should cost more than 3-star"

    def test_peak_season_costs_more_than_off_peak(self, trained_pipeline):
        """Sanity check: peak season prices should exceed off-peak."""
        pipeline, _, _ = trained_pipeline
        df = generate_hotel_data(n_samples=200, random_state=0)
        base = df.iloc[:20].copy()

        peak = base.copy()
        peak["season"] = "peak"
        off_peak = base.copy()
        off_peak["season"] = "off-peak"

        pred_peak = pipeline.predict(peak[FEATURE_COLUMNS]).mean()
        pred_off = pipeline.predict(off_peak[FEATURE_COLUMNS]).mean()
        assert pred_peak > pred_off, "Peak season should be more expensive than off-peak"
