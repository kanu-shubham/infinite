"""Feature schema and vectorization for the TravelAds ranker.

The serving path constructs a numpy matrix of shape (n_candidates, N_FEATURES)
without per-row Python overhead. Categorical features are hashed mod a fixed
vocab size; numeric features are passed through (LightGBM handles NaN natively).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# Hashed-categorical vocab sizes. Powers of two for fast modulo.
COUNTRY_BUCKETS = 256
DEVICE_BUCKETS = 8
SEGMENT_BUCKETS = 64
DEST_BUCKETS = 1024
ADVERTISER_BUCKETS = 4096

FEATURE_NAMES = [
    # user
    "u_country_h", "u_device_h", "u_segment_h", "u_recent_bookings",
    # ad / hotel
    "a_star", "a_price_tier", "a_country_h", "a_advertiser_h",
    "a_age_hours", "a_hist_ctr", "a_hist_cvr", "a_quality",
    # context
    "c_dest_h", "c_lead_time", "c_los", "c_pax",
    "c_hour", "c_dow", "c_is_weekend",
    # cross
    "x_user_ad_country_match", "x_bid_cpc", "x_bid_x_ctr",
]
N_FEATURES = len(FEATURE_NAMES)


def _h(s: str | None, mod: int) -> int:
    if not s:
        return 0
    # Stable, fast string hash (Python's hash() is salted per-process; use FNV-1a).
    h = 0xcbf29ce484222325
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001b3) & 0xffffffffffffffff
    return h % mod


@dataclass(slots=True)
class UserFeatures:
    country: str
    device: str
    segment: str = "unknown"
    recent_bookings: float = 0.0


@dataclass(slots=True)
class AdFeatures:
    star: float
    price_tier: float
    country: str
    advertiser_id: str
    age_hours: float
    hist_ctr: float
    hist_cvr: float
    quality: float


@dataclass(slots=True)
class Context:
    destination: str
    lead_time_days: float
    los: float
    pax: float
    hour: int
    dow: int


def vectorize(
    user: UserFeatures,
    ads: Sequence[AdFeatures],
    bids: np.ndarray,
    ctx: Context,
) -> np.ndarray:
    """Return float32 (n, N_FEATURES) matrix. No per-candidate Python objects in hot loop."""
    n = len(ads)
    X = np.empty((n, N_FEATURES), dtype=np.float32)

    u_country = _h(user.country, COUNTRY_BUCKETS)
    u_device = _h(user.device, DEVICE_BUCKETS)
    u_segment = _h(user.segment, SEGMENT_BUCKETS)
    c_dest = _h(ctx.destination, DEST_BUCKETS)
    is_weekend = 1.0 if ctx.dow >= 5 else 0.0

    for i, ad in enumerate(ads):
        a_country = _h(ad.country, COUNTRY_BUCKETS)
        row = X[i]
        row[0] = u_country
        row[1] = u_device
        row[2] = u_segment
        row[3] = user.recent_bookings
        row[4] = ad.star
        row[5] = ad.price_tier
        row[6] = a_country
        row[7] = _h(ad.advertiser_id, ADVERTISER_BUCKETS)
        row[8] = ad.age_hours
        row[9] = ad.hist_ctr
        row[10] = ad.hist_cvr
        row[11] = ad.quality
        row[12] = c_dest
        row[13] = ctx.lead_time_days
        row[14] = ctx.los
        row[15] = ctx.pax
        row[16] = ctx.hour
        row[17] = ctx.dow
        row[18] = is_weekend
        row[19] = 1.0 if a_country == u_country else 0.0
        row[20] = bids[i]
        row[21] = bids[i] * ad.hist_ctr
    return X
