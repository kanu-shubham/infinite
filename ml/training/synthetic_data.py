"""Generate synthetic TravelAds impressions with click labels.

The signal embedded in the data:
  * higher star rating + matching country + lower price tier => higher click prob
  * weekend / longer LOS shifts preferences toward higher-star
  * hist_ctr is a noisy but strong signal (proxy for collaborative filtering)

This lets the trained model exhibit realistic feature importances when inspected.
"""
from __future__ import annotations

import numpy as np

COUNTRIES = ["US", "GB", "FR", "DE", "IN", "JP", "BR", "ES", "IT", "AU"]
DEVICES = ["mobile", "desktop", "tablet"]
SEGMENTS = ["new", "returning", "loyal", "business", "leisure"]
DESTINATIONS = ["PAR", "NYC", "LON", "TYO", "BCN", "ROM", "BER", "SYD", "BKK", "RIO"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(n_rows: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X, y) where X is (n, N_FEATURES) float32 and y is 0/1 click labels."""
    from ml.models.features import (  # local import keeps training optional in serving image
        N_FEATURES,
        UserFeatures,
        AdFeatures,
        Context,
        vectorize,
    )

    rng = np.random.default_rng(seed)
    X = np.empty((n_rows, N_FEATURES), dtype=np.float32)
    logits = np.empty(n_rows, dtype=np.float32)

    # Batch by impression of ~30 candidates; each row is one (user, ad, ctx).
    i = 0
    while i < n_rows:
        n = min(30, n_rows - i)
        u = UserFeatures(
            country=rng.choice(COUNTRIES),
            device=rng.choice(DEVICES),
            segment=rng.choice(SEGMENTS),
            recent_bookings=float(rng.integers(0, 8)),
        )
        ctx = Context(
            destination=rng.choice(DESTINATIONS),
            lead_time_days=float(rng.integers(0, 90)),
            los=float(rng.integers(1, 14)),
            pax=float(rng.integers(1, 5)),
            hour=int(rng.integers(0, 24)),
            dow=int(rng.integers(0, 7)),
        )
        ads = []
        for _ in range(n):
            ads.append(
                AdFeatures(
                    star=float(rng.integers(1, 6)),
                    price_tier=float(rng.integers(1, 6)),
                    country=rng.choice(COUNTRIES),
                    advertiser_id=f"adv_{rng.integers(0, 500)}",
                    age_hours=float(rng.exponential(48)),
                    hist_ctr=float(np.clip(rng.beta(2, 50), 0, 1)),
                    hist_cvr=float(np.clip(rng.beta(2, 100), 0, 1)),
                    quality=float(np.clip(rng.normal(0.7, 0.15), 0.1, 1.0)),
                )
            )
        bids = rng.uniform(0.2, 3.0, size=n).astype(np.float32)
        X[i : i + n] = vectorize(u, ads, bids, ctx)

        # Click signal: linear combo of meaningful columns + noise.
        stars = np.array([a.star for a in ads])
        price_tier = np.array([a.price_tier for a in ads])
        hist_ctr = np.array([a.hist_ctr for a in ads])
        country_match = np.array([1.0 if a.country == u.country else 0.0 for a in ads])
        logits[i : i + n] = (
            -4.2
            + 0.35 * stars
            - 0.20 * price_tier
            + 8.0 * hist_ctr
            + 0.6 * country_match
            + 0.05 * (ctx.los if ctx.dow >= 5 else 0)
            + rng.normal(0, 0.4, size=n)
        )
        i += n

    p = _sigmoid(logits)
    y = (rng.uniform(size=n_rows) < p).astype(np.int8)
    return X, y
