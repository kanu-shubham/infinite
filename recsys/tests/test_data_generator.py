"""Tests for the synthetic data generator."""
import pandas as pd
import pytest

from src.data_generator import (
    generate_dataset,
    generate_hotel_features,
    generate_user_features,
    generate_interactions,
)
import numpy as np


RNG = np.random.RandomState(0)


class TestUserFeatures:
    def test_correct_row_count(self):
        df = generate_user_features(100, RNG)
        assert len(df) == 100

    def test_no_nulls(self):
        df = generate_user_features(200, RNG)
        assert df.isnull().sum().sum() == 0

    def test_price_sensitivity_in_range(self):
        df = generate_user_features(500, RNG)
        assert df["price_sensitivity"].between(0.0, 1.0).all()


class TestHotelFeatures:
    def test_correct_row_count(self):
        df = generate_hotel_features(50, RNG)
        assert len(df) == 50

    def test_star_rating_valid(self):
        df = generate_hotel_features(200, RNG)
        assert df["star_rating"].isin([3, 4, 5]).all()

    def test_no_nulls(self):
        df = generate_hotel_features(100, RNG)
        assert df.isnull().sum().sum() == 0


class TestInteractions:
    def test_labels_are_binary(self):
        users = generate_user_features(50, RNG)
        hotels = generate_hotel_features(20, RNG)
        ints = generate_interactions(users, hotels, 200, RNG)
        assert ints["label"].isin([0, 1]).all()

    def test_ratings_in_range(self):
        users = generate_user_features(50, RNG)
        hotels = generate_hotel_features(20, RNG)
        ints = generate_interactions(users, hotels, 200, RNG)
        assert ints["rating"].between(1.0, 5.0).all()


class TestGenerateDataset:
    def test_returns_three_dataframes(self):
        users, hotels, ints = generate_dataset()
        assert isinstance(users, pd.DataFrame)
        assert isinstance(hotels, pd.DataFrame)
        assert isinstance(ints, pd.DataFrame)

    def test_positive_interaction_rate(self):
        _, _, ints = generate_dataset()
        # Should have a meaningful number of positives
        assert 0.1 < ints["label"].mean() < 0.9
