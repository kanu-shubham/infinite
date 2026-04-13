"""
Tests for the data generation and preprocessing pipeline.

These run fast (no model training) and verify the data contract
that the rest of the system depends on.
"""
import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    build_preprocessor,
    generate_hotel_data,
)


class TestGenerateHotelData:
    def test_returns_correct_row_count(self):
        df = generate_hotel_data(n_samples=200)
        assert len(df) == 200

    def test_all_expected_columns_present(self):
        df = generate_hotel_data(n_samples=50)
        for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
            assert col in df.columns, f"Missing column: {col}"

    def test_price_is_always_positive(self):
        df = generate_hotel_data(n_samples=500)
        assert (df[TARGET_COLUMN] > 0).all(), "Found non-positive prices"

    def test_star_rating_only_valid_values(self):
        df = generate_hotel_data(n_samples=500)
        assert df["star_rating"].isin([3, 4, 5]).all()

    def test_review_score_in_range(self):
        df = generate_hotel_data(n_samples=500)
        assert df["review_score"].between(1.0, 10.0).all()

    def test_distance_in_range(self):
        df = generate_hotel_data(n_samples=500)
        assert df["distance_to_center_km"].between(0.0, 30.0).all()

    def test_reproducibility_with_same_seed(self):
        df1 = generate_hotel_data(n_samples=100, random_state=7)
        df2 = generate_hotel_data(n_samples=100, random_state=7)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_data(self):
        df1 = generate_hotel_data(n_samples=100, random_state=1)
        df2 = generate_hotel_data(n_samples=100, random_state=2)
        assert not df1[TARGET_COLUMN].equals(df2[TARGET_COLUMN])

    def test_no_null_values(self):
        df = generate_hotel_data(n_samples=300)
        assert df.isnull().sum().sum() == 0, "Unexpected null values"


class TestPreprocessor:
    def test_preprocessor_transforms_without_error(self):
        df = generate_hotel_data(n_samples=100)
        preprocessor = build_preprocessor()
        result = preprocessor.fit_transform(df[FEATURE_COLUMNS])
        assert result.shape[0] == 100

    def test_output_has_no_nans(self):
        df = generate_hotel_data(n_samples=100)
        preprocessor = build_preprocessor()
        result = preprocessor.fit_transform(df[FEATURE_COLUMNS])
        assert not np.isnan(result).any()

    def test_numerical_features_are_scaled(self):
        df = generate_hotel_data(n_samples=1000)
        preprocessor = build_preprocessor()
        result = preprocessor.fit_transform(df[FEATURE_COLUMNS])
        # Scaled numerical columns should have mean ≈ 0
        n_num = len(NUMERICAL_FEATURES)
        numerical_output = result[:, :n_num]
        assert abs(numerical_output.mean()) < 0.1
