"""
Feature engineering — builds the (user, item, interaction) feature matrix
that feeds the XGBoost ranking model.

For each (user, candidate_product) pair this module computes:
  • All 53 user features  (from UserProfile.to_feature_vector)
  • All 50 item features  (from InsuranceProduct.to_feature_vector)
  • 7  interaction features derived from combining both

Total ranking feature vector: 53 + 50 + 7 = 110 dims

Interaction features:
  [0] price_delta          maxPricePerDay - productPrice  (raw)
  [1] price_fit            sigmoid(2 * price_delta)
  [2] coverage_ratio       log1p(cov/minCov) / log1p(5)  clipped to [0,1]
  [3] feature_overlap      |wanted ∩ has| / |wanted|
  [4] trip_length_ok       1.0 if maxTripDays >= duration else 0.0
  [5] ann_score            cosine similarity from retrieval stage (0 if unavailable)
  [6] destination_overlap  |user_dest ∩ item_dest| / |user_dest|
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .catalogue import InsuranceProduct, ITEM_FEATURE_DIM
from .interaction_log import UserProfile, USER_FEATURE_DIM

INTERACTION_FEATURE_DIM = 7
RANKING_FEATURE_DIM     = USER_FEATURE_DIM + ITEM_FEATURE_DIM + INTERACTION_FEATURE_DIM  # 110

RANKING_FEATURE_NAMES = (
    [f"user_{i}" for i in range(USER_FEATURE_DIM)]
    + [f"item_{i}" for i in range(ITEM_FEATURE_DIM)]
    + [
        "price_delta", "price_fit", "coverage_ratio",
        "feature_overlap", "trip_length_ok", "ann_score",
        "destination_overlap",
    ]
)


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def compute_interaction_features(
    user: UserProfile,
    product: InsuranceProduct,
    ann_score: float = 0.0,
) -> np.ndarray:
    """Return the 7-dim interaction feature vector for one (user, item) pair."""
    v = np.zeros(INTERACTION_FEATURE_DIM, dtype=np.float32)

    # [0] Raw price delta
    price_delta = user.max_price_per_day - product.price_per_day
    v[0] = float(price_delta)

    # [1] Sigmoid price fit — ~1 when product is cheap, drops above budget
    v[1] = float(_sigmoid(2.0 * price_delta))

    # [2] Coverage ratio (log-compressed)
    cov_ratio = product.coverage_amount / max(user.min_coverage, 1.0)
    v[2] = float(np.clip(np.log1p(cov_ratio) / np.log1p(5.0), 0.0, 1.0))

    # [3] Feature overlap fraction
    wanted = set(user.feature_indices)
    has    = set(product.feature_indices)
    v[3] = len(wanted & has) / max(len(wanted), 1)

    # [4] Trip length gate
    v[4] = 1.0 if product.max_trip_days >= user.trip_duration else 0.0

    # [5] ANN cosine score carried from retrieval stage
    v[5] = float(np.clip(ann_score, -1.0, 1.0))

    # [6] Destination overlap
    u_dest = set(user.destination_indices)
    p_dest = set(product.destination_indices)
    v[6] = len(u_dest & p_dest) / max(len(u_dest), 1)

    return v


def build_ranking_row(
    user: UserProfile,
    product: InsuranceProduct,
    ann_score: float = 0.0,
) -> np.ndarray:
    """Concatenate user / item / interaction features into a single 110-dim row."""
    return np.concatenate([
        user.to_feature_vector(),
        product.to_feature_vector(),
        compute_interaction_features(user, product, ann_score),
    ])


def build_ranking_dataset(
    users_by_id: dict[str, UserProfile],
    products_by_id: dict[str, InsuranceProduct],
    interactions: pd.DataFrame,
    label_col: str = "purchased",
    neg_sample_ratio: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) for the ranking model from the interaction log.

    Positive examples: rows where label_col == 1.
    Negative examples: for each positive, sample neg_sample_ratio random
      products the user did NOT click (hard negatives are more informative
      than random catalogue negatives, but here random is sufficient).

    Returns:
      X : float32 array of shape (n_samples, 110)
      y : int32 array of shape (n_samples,), values in {0, 1}
    """
    rng = np.random.default_rng(seed)
    all_product_ids = list(products_by_id.keys())

    rows_X: list[np.ndarray] = []
    rows_y: list[int]        = []

    positives = interactions[interactions[label_col] == 1]

    for _, pos_row in positives.iterrows():
        uid  = pos_row["user_id"]
        pid  = pos_row["product_id"]
        user = users_by_id.get(uid)
        prod = products_by_id.get(pid)
        if user is None or prod is None:
            continue

        # Positive example
        rows_X.append(build_ranking_row(user, prod, ann_score=0.9))
        rows_y.append(1)

        # Sample negatives: products the user didn't purchase
        clicked_ids = set(
            interactions[(interactions["user_id"] == uid) & (interactions[label_col] == 1)]["product_id"]
        )
        neg_pool = [p for p in all_product_ids if p not in clicked_ids]
        if not neg_pool:
            continue
        n_neg = min(neg_sample_ratio, len(neg_pool))
        for neg_pid in rng.choice(neg_pool, size=n_neg, replace=False):
            neg_prod = products_by_id[neg_pid]
            rows_X.append(build_ranking_row(user, neg_prod, ann_score=float(rng.uniform(0.3, 0.7))))
            rows_y.append(0)

    X = np.vstack(rows_X).astype(np.float32)
    y = np.array(rows_y, dtype=np.int32)
    print(f"Ranking dataset: {X.shape[0]:,} rows | positives={y.sum():,} ({y.mean():.2%})")
    return X, y
