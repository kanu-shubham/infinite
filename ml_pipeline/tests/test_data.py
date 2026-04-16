"""
Unit tests for the data generation module.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from data.generate_data import generate_hotels


@pytest.fixture(scope="module")
def df():
    return generate_hotels(n=500, seed=0)


def test_row_count(df):
    assert len(df) == 500


def test_required_columns(df):
    required = {"hotel_id", "city", "category", "star_rating",
                 "review_score", "price_usd", "event_timestamp"}
    assert required.issubset(df.columns)


def test_no_nulls(df):
    assert df.isnull().sum().sum() == 0, "Dataset contains unexpected nulls"


def test_price_range(df):
    assert df["price_usd"].min() >= 20
    assert df["price_usd"].max() <= 1500


def test_star_rating_values(df):
    assert set(df["star_rating"].unique()).issubset({1, 2, 3, 4, 5})


def test_review_score_range(df):
    assert df["review_score"].between(1, 10).all()


def test_deterministic(df):
    """Same seed must produce the same data every time."""
    df2 = generate_hotels(n=500, seed=0)
    pd.testing.assert_frame_equal(df.reset_index(drop=True),
                                   df2.reset_index(drop=True))
