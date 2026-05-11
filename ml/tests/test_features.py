import numpy as np

from ml.models.features import (
    AdFeatures,
    Context,
    N_FEATURES,
    UserFeatures,
    vectorize,
)


def _make_ads(n: int) -> list[AdFeatures]:
    return [
        AdFeatures(
            star=3.0,
            price_tier=2.0,
            country="US",
            advertiser_id=f"adv_{i}",
            age_hours=12.0,
            hist_ctr=0.05,
            hist_cvr=0.01,
            quality=0.8,
        )
        for i in range(n)
    ]


def test_vectorize_shape_and_dtype():
    u = UserFeatures(country="US", device="mobile", segment="loyal", recent_bookings=2)
    ctx = Context(destination="PAR", lead_time_days=10, los=3, pax=2, hour=14, dow=5)
    ads = _make_ads(50)
    bids = np.full(50, 1.25, dtype=np.float32)
    X = vectorize(u, ads, bids, ctx)
    assert X.shape == (50, N_FEATURES)
    assert X.dtype == np.float32


def test_country_match_signal():
    u = UserFeatures(country="US", device="mobile")
    ctx = Context(destination="NYC", lead_time_days=3, los=2, pax=1, hour=9, dow=2)
    ads = [
        AdFeatures(3, 2, "US", "a", 1, 0.02, 0.005, 0.7),
        AdFeatures(3, 2, "JP", "b", 1, 0.02, 0.005, 0.7),
    ]
    X = vectorize(u, ads, np.array([1.0, 1.0], dtype=np.float32), ctx)
    # x_user_ad_country_match is column 19 in the FEATURE_NAMES order.
    assert X[0, 19] == 1.0
    assert X[1, 19] == 0.0
