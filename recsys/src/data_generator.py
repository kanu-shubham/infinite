"""
data_generator.py
-----------------
Generates a synthetic user-hotel interaction dataset for training
the two-tower recommendation model.

In production this would come from:
  - A data warehouse (BigQuery / Snowflake) via SQL query
  - A feature store (Feast, Tecton) for pre-computed user/item features
  - A Kafka stream of real-time click/booking events

Schema produced:
  interactions  — (user_id, hotel_id, rating, timestamp)
  user_features — (user_id, age_group, travel_type, price_sensitivity, ...)
  hotel_features— (hotel_id, city, star_rating, avg_review, price_tier, ...)
"""
import logging

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


CITIES = ["New York", "London", "Paris", "Tokyo", "Dubai", "Singapore", "Sydney"]
TRAVEL_TYPES = ["business", "leisure", "family", "solo"]
AGE_GROUPS = ["18-25", "26-35", "36-50", "51+"]
PRICE_TIERS = ["budget", "mid", "luxury"]


def generate_user_features(n_users: int, rng: np.random.RandomState) -> pd.DataFrame:
    """Each user has stable demographic and preference features."""
    return pd.DataFrame(
        {
            "user_id": range(n_users),
            "age_group": rng.choice(AGE_GROUPS, n_users),
            "travel_type": rng.choice(TRAVEL_TYPES, n_users),
            # How sensitive this user is to price (0=not at all, 1=very much)
            "price_sensitivity": np.round(rng.uniform(0.0, 1.0, n_users), 2),
            # How much they care about reviews
            "review_weight": np.round(rng.uniform(0.0, 1.0, n_users), 2),
            # Preferred destination type
            "prefers_city_center": rng.randint(0, 2, n_users),
            "loyalty_tier": rng.choice(["bronze", "silver", "gold"], n_users),
        }
    )


def generate_hotel_features(n_hotels: int, rng: np.random.RandomState) -> pd.DataFrame:
    """Each hotel has stable attributes used as item features."""
    return pd.DataFrame(
        {
            "hotel_id": range(n_hotels),
            "city": rng.choice(CITIES, n_hotels),
            "star_rating": rng.choice([3, 4, 5], n_hotels),
            "avg_review_score": np.round(rng.uniform(6.5, 9.8, n_hotels), 1),
            "price_tier": rng.choice(PRICE_TIERS, n_hotels),
            "distance_to_center_km": np.round(rng.uniform(0.2, 12.0, n_hotels), 1),
            "amenities_count": rng.randint(5, 40, n_hotels),
            "has_pool": rng.randint(0, 2, n_hotels),
            "has_spa": rng.randint(0, 2, n_hotels),
            "has_gym": rng.randint(0, 2, n_hotels),
        }
    )


def generate_interactions(
    user_df: pd.DataFrame,
    hotel_df: pd.DataFrame,
    n_interactions: int,
    rng: np.random.RandomState,
) -> pd.DataFrame:
    """
    Generate user-hotel interactions (implicit + explicit feedback).

    Ratings are NOT random — they are biased by user preferences vs hotel
    attributes, so the model has a real signal to learn.
    """
    user_ids = rng.randint(0, len(user_df), n_interactions)
    hotel_ids = rng.randint(0, len(hotel_df), n_interactions)

    users = user_df.iloc[user_ids].reset_index(drop=True)
    hotels = hotel_df.iloc[hotel_ids].reset_index(drop=True)

    # Build rating from alignment between user preferences and hotel attributes
    price_tiers_num = {"budget": 1, "mid": 2, "luxury": 3}
    loyalty_bonus = {"bronze": 0, "silver": 0.2, "gold": 0.4}

    review_bonus = (hotels["avg_review_score"] - 7.0) * users["review_weight"]
    price_alignment = -(
        users["price_sensitivity"]
        * hotels["price_tier"].map(price_tiers_num)
    )
    location_bonus = (
        users["prefers_city_center"]
        * (1.0 - hotels["distance_to_center_km"] / 12.0)
    )
    star_bonus = (hotels["star_rating"] - 3) * 0.3
    loyalty_b = users["loyalty_tier"].map(loyalty_bonus)
    noise = rng.normal(0, 0.5, n_interactions)

    raw_rating = 3.5 + review_bonus + price_alignment + location_bonus + star_bonus + loyalty_b + noise
    rating = np.clip(np.round(raw_rating, 1), 1.0, 5.0)

    return pd.DataFrame(
        {
            "user_id": user_ids,
            "hotel_id": hotel_ids,
            "rating": rating,
            # 1 if rating >= 4 (positive interaction), used for retrieval training
            "label": (rating >= 4.0).astype(int),
        }
    )


def generate_dataset(config_path: str = "config.yaml") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (user_features, hotel_features, interactions)."""
    config = load_config(config_path)
    cfg = config["data"]
    rng = np.random.RandomState(cfg["random_state"])

    logger.info(f"Generating dataset: {cfg['n_users']} users, "
                f"{cfg['n_hotels']} hotels, {cfg['n_interactions']} interactions")

    users = generate_user_features(cfg["n_users"], rng)
    hotels = generate_hotel_features(cfg["n_hotels"], rng)
    interactions = generate_interactions(users, hotels, cfg["n_interactions"], rng)

    pos_rate = interactions["label"].mean()
    logger.info(f"Positive interaction rate: {pos_rate:.1%}")

    return users, hotels, interactions
