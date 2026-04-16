"""
generate_data.py
----------------
Creates a synthetic hotel dataset that the entire ML pipeline uses.

WHY synthetic data?
  Real hotel pricing data costs money and has privacy concerns.
  Synthetic data lets you learn every pipeline step without blockers.
  Once you understand the pipeline, swapping in real data is trivial.

The output is a CSV file:  data/hotels.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_HOTELS = 2_000   # enough to train a decent model


def generate_hotels(n: int = N_HOTELS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ── Categorical features ──────────────────────────────────
    cities = ["New York", "Paris", "Tokyo", "London", "Sydney",
              "Dubai", "Singapore", "Barcelona", "Rome", "Bangkok"]
    categories = ["Budget", "Mid-range", "Luxury", "Resort", "Boutique"]

    city           = rng.choice(cities, n)
    category       = rng.choice(categories, n)

    # ── Numeric features ──────────────────────────────────────
    star_rating    = rng.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.10, 0.30, 0.35, 0.20])
    review_score   = np.clip(rng.normal(loc=7.5, scale=1.2, size=n), 1, 10).round(1)
    num_reviews    = rng.integers(10, 5000, n)
    distance_km    = np.clip(rng.exponential(scale=3.0, size=n), 0.1, 30).round(2)
    amenities      = rng.integers(0, 20, n)   # count of amenities (pool, gym, …)
    rooms_available= rng.integers(1, 50, n)

    # ── Target: price per night (USD) ─────────────────────────
    # We build a realistic price that depends on the features above.
    # Noise is added so the model has something to learn beyond a formula.
    base_price = (
          star_rating   * 30
        + review_score  * 8
        + amenities     * 4
        - distance_km   * 3
    )
    city_premium = dict(zip(cities, [80, 60, 40, 70, 50, 100, 55, 45, 35, 25]))
    cat_premium  = {"Budget": -40, "Mid-range": 0, "Luxury": 120, "Resort": 90, "Boutique": 50}

    price = (
          base_price
        + np.array([city_premium[c] for c in city])
        + np.array([cat_premium[c]  for c in category])
        + rng.normal(0, 20, n)      # random noise
    ).clip(20, 1500).round(2)

    # ── Timestamps (needed by Feast for point-in-time joins) ──
    # Each row represents a hotel "snapshot" taken on a random date.
    start = pd.Timestamp("2023-01-01")
    end   = pd.Timestamp("2024-01-01")
    event_timestamps = pd.to_datetime(
        rng.integers(start.value, end.value, n)
    )

    df = pd.DataFrame({
        "hotel_id":         range(1, n + 1),
        "city":             city,
        "category":         category,
        "star_rating":      star_rating,
        "review_score":     review_score,
        "num_reviews":      num_reviews,
        "distance_km":      distance_km,
        "amenities":        amenities,
        "rooms_available":  rooms_available,
        "price_usd":        price,           # TARGET
        "event_timestamp":  event_timestamps,
    })

    return df


if __name__ == "__main__":
    out_path = Path(__file__).parent / "hotels.csv"
    df = generate_hotels()
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df):,} hotels → {out_path}")
    print(df.describe().to_string())
