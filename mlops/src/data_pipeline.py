"""
data_pipeline.py
----------------
Generates synthetic hotel data and builds the sklearn preprocessing pipeline.

In a real production system this module would:
  - Pull raw data from a data warehouse (Snowflake, BigQuery, etc.)
  - Apply feature engineering
  - Validate schema with Great Expectations or Pandera
  - Log a data snapshot to the MLflow artifact store

For this tutorial we use synthetic data so you can run everything locally
without external dependencies.
"""
import logging

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Feature metadata ─────────────────────────────────────────────────────────

CATEGORICAL_FEATURES = ["city", "season"]
NUMERICAL_FEATURES = [
    "star_rating",
    "amenities_count",
    "distance_to_center_km",
    "review_score",
    "rooms_available",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
TARGET_COLUMN = "price_per_night"

CITIES = ["New York", "London", "Paris", "Tokyo", "Dubai"]
SEASONS = ["peak", "off-peak", "shoulder"]


# ── Data generation ───────────────────────────────────────────────────────────

def generate_hotel_data(n_samples: int = 10_000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic hotel dataset.

    Price is a deterministic function of the features plus Gaussian noise,
    so a well-trained model should achieve R2 > 0.95 on held-out data.
    """
    rng = np.random.RandomState(random_state)

    city_base_price = {
        "New York": 220,
        "London": 190,
        "Paris": 170,
        "Tokyo": 155,
        "Dubai": 230,
    }
    season_multiplier = {"peak": 1.40, "shoulder": 1.00, "off-peak": 0.72}

    data = {
        "city": rng.choice(CITIES, n_samples),
        "star_rating": rng.choice([3, 4, 5], n_samples),
        "amenities_count": rng.randint(5, 31, n_samples),
        "distance_to_center_km": np.round(rng.uniform(0.1, 15.0, n_samples), 2),
        "review_score": np.round(rng.uniform(6.0, 10.0, n_samples), 1),
        "season": rng.choice(SEASONS, n_samples),
        "rooms_available": rng.randint(1, 51, n_samples),
    }
    df = pd.DataFrame(data)

    base = df["city"].map(city_base_price)
    season_effect = base * (df["season"].map(season_multiplier) - 1.0)
    star_effect = (df["star_rating"] - 3) * 85
    amenity_effect = df["amenities_count"] * 3.5
    distance_penalty = df["distance_to_center_km"] * 6
    review_effect = (df["review_score"] - 8.0) * 22
    noise = rng.normal(0, 18, n_samples)

    raw_price = base + season_effect + star_effect + amenity_effect - distance_penalty + review_effect + noise
    df[TARGET_COLUMN] = np.round(np.maximum(raw_price, 50.0), 2)

    logger.info(f"Generated {n_samples} hotel records. Price range: "
                f"${df[TARGET_COLUMN].min():.0f} – ${df[TARGET_COLUMN].max():.0f}")
    return df


# ── Preprocessing pipeline ────────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """
    Returns a fitted-ready ColumnTransformer:
      - StandardScaler  on numerical columns
      - OneHotEncoder   on categorical columns
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
