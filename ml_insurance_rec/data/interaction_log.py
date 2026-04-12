"""
Interaction log — simulates real user-product event data.

Generates N_USERS synthetic user profiles, then for each user runs
M_SESSIONS sessions where the user is shown a random slate of products.
A click is sampled from a logistic model that rewards type match,
price fit, coverage fit, feature overlap and trip-length compatibility.
A purchase is sampled conditional on click.

Output schema (one row per impression):
  user_id, product_id, session_id,
  clicked (0/1), purchased (0/1),
  timestamp (unix seconds, ordered)

These (user, item, label) triplets are the raw training signal for both
the two-tower model and the ranking model.

User feature vector layout — 53 dims:
  [0]     age normalised             (/ 80)
  [1]     trip_duration normalised   (/ 90)
  [2]     price_sensitivity          (maxPricePerDay / 20)
  [3]     min_coverage log-norm      (log1p / log1p(1e6))
  [4:12]  primary type one-hot       8
  [12:20] secondary type one-hot     8
  [20:29] destination flags          9
  [29:45] feature flags             16
  [45:53] style flags                8
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from .catalogue import (
    TYPES, FEATURES, DESTINATIONS, STYLES,
    FEATURES_BY_TYPE, DESTINATIONS_BY_TYPE, STYLES_BY_TYPE,
    PRICE_RANGE, InsuranceProduct,
)

USER_FEATURE_DIM = 53

# ── Simulation hyper-parameters ───────────────────────────────────────────────
N_USERS       = 2_000   # synthetic users
N_SESSIONS    = 15      # sessions per user
SLATE_SIZE    = 10      # products shown per session
PURCHASE_RATE = 0.25    # P(purchase | click)


# ── User profile ──────────────────────────────────────────────────────────────

@dataclass
class UserProfile:
    user_id: str
    age: float                      # 18-80
    trip_duration: int              # days
    max_price_per_day: float
    min_coverage: float
    preferred_type_idx: int         # primary type (index into TYPES)
    secondary_type_idx: int
    destination_indices: List[int]  # desired destinations
    feature_indices: List[int]      # desired features (subset of 0-15)
    style_indices: List[int]        # travel styles
    # Cached feature vector
    _fv: np.ndarray = field(default=None, repr=False, compare=False)

    def to_feature_vector(self) -> np.ndarray:
        if self._fv is not None:
            return self._fv
        v = np.zeros(USER_FEATURE_DIM, dtype=np.float32)
        v[0] = self.age / 80.0
        v[1] = min(self.trip_duration, 90) / 90.0
        v[2] = self.max_price_per_day / 20.0
        v[3] = np.log1p(self.min_coverage) / np.log1p(1_000_000)
        # [4:12] primary type one-hot
        v[4 + self.preferred_type_idx] = 1.0
        # [12:20] secondary type, softer signal
        v[12 + self.secondary_type_idx] = 0.6
        # [20:29] destination flags
        for di in self.destination_indices:
            if 0 <= di < 9:
                v[20 + di] = 1.0
        # [29:45] feature flags (top-2 weighted higher)
        for rank, fi in enumerate(self.feature_indices):
            if 0 <= fi < 16:
                v[29 + fi] = 1.0 if rank < 2 else 0.7
        # [45:53] style flags
        for si in self.style_indices:
            if 0 <= si < 8:
                v[45 + si] = 1.0
        self._fv = v
        return v

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "age": self.age,
            "trip_duration": self.trip_duration,
            "max_price_per_day": self.max_price_per_day,
            "min_coverage": self.min_coverage,
            "preferred_type_idx": self.preferred_type_idx,
            "secondary_type_idx": self.secondary_type_idx,
            "destination_indices": self.destination_indices,
            "feature_indices": self.feature_indices,
            "style_indices": self.style_indices,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UserProfile":
        return cls(**{k: v for k, v in d.items() if k != "_fv"})


# ── User generator ────────────────────────────────────────────────────────────

# Persona archetypes: (preferred_type, secondary_type, age_range, dur_range, price_range, min_cov)
_ARCHETYPES = [
    ("adventure",     "medical",       (22, 40), (7,  21), (8,  15), 100_000),
    ("cfar",          "comprehensive", (30, 55), (5,  14), (12, 20), 50_000),
    ("family",        "comprehensive", (28, 50), (7,  21), (6,  12), 30_000),
    ("student",       "basic",         (18, 28), (14, 90), (1,   4), 15_000),
    ("senior",        "medical",       (55, 80), (14, 30), (8,  18), 150_000),
    ("basic",         "cfar",          (20, 45), (3,   7), (2,   5), 10_000),
    ("comprehensive", "cfar",          (30, 60), (7,  21), (8,  15), 75_000),
    ("medical",       "senior",        (50, 80), (14, 30), (5,  10), 200_000),
]


def _generate_users(n: int, rng: np.random.Generator) -> List[UserProfile]:
    users = []
    for i in range(n):
        arch = _ARCHETYPES[i % len(_ARCHETYPES)]
        ptype, stype, age_r, dur_r, price_r, min_cov = arch

        pidx = TYPES.index(ptype)
        sidx = TYPES.index(stype)

        dest_base  = list(DESTINATIONS_BY_TYPE[ptype])
        feat_base  = list(FEATURES_BY_TYPE[ptype])
        style_base = list(STYLES_BY_TYPE[ptype])

        # Add a little random variation per user
        extra_dest  = [int(rng.integers(9))]
        extra_feat  = [int(rng.integers(16))]

        users.append(UserProfile(
            user_id             = f"user_{i:05d}",
            age                 = float(rng.integers(age_r[0], age_r[1] + 1)),
            trip_duration       = int(rng.integers(dur_r[0], dur_r[1] + 1)),
            max_price_per_day   = round(float(rng.uniform(price_r[0], price_r[1])), 2),
            min_coverage        = float(min_cov) * rng.uniform(0.7, 1.3),
            preferred_type_idx  = pidx,
            secondary_type_idx  = sidx,
            destination_indices = list(set(dest_base + extra_dest)),
            feature_indices     = list(set(feat_base + extra_feat))[:8],
            style_indices       = style_base,
        ))
    return users


# ── Click probability model ───────────────────────────────────────────────────

def _click_logit(user: UserProfile, product: InsuranceProduct) -> float:
    """
    Log-odds of a click.  Mirrors the signals the ranking model will learn.
    Positive terms push toward click; bias keeps base rate low (~8-15%).
    """
    # Type match — strongest signal
    type_match = float(TYPES[user.preferred_type_idx] == product.type) * 3.0
    type_match += float(TYPES[user.secondary_type_idx] == product.type) * 1.5

    # Price fit — sigmoid-like: reward if price <= max, penalise above
    price_delta = user.max_price_per_day - product.price_per_day
    price_fit   = float(np.clip(price_delta / user.max_price_per_day, -1, 1)) * 1.5

    # Coverage fit
    cov_ratio  = product.coverage_amount / max(user.min_coverage, 1)
    cov_fit    = float(np.clip(np.log1p(cov_ratio) / np.log1p(5), 0, 1)) * 1.0

    # Feature overlap — fraction of wanted features present
    wanted = set(user.feature_indices)
    has    = set(product.feature_indices)
    feat_overlap = len(wanted & has) / max(len(wanted), 1) * 2.0

    # Trip length hard gate
    trip_ok = 1.0 if product.max_trip_days >= user.trip_duration else -2.0

    # Provider quality proxy
    quality = (product.rating / 5.0 + product.claim_score) / 2.0 * 0.5

    # Freshness small boost
    freshness = product.freshness_score * 0.3

    return type_match + price_fit + cov_fit + feat_overlap + trip_ok + quality + freshness - 5.5


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


# ── Interaction log generator ─────────────────────────────────────────────────

def generate_interactions(
    users: List[UserProfile],
    products: List[InsuranceProduct],
    n_sessions: int = N_SESSIONS,
    slate_size: int = SLATE_SIZE,
    purchase_rate: float = PURCHASE_RATE,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Simulate user sessions and return a DataFrame of impression events.

    Returns columns:
      user_id, product_id, session_id, clicked, purchased, timestamp
    """
    rng    = np.random.default_rng(seed)
    rows   = []
    t_base = int(time.time()) - n_sessions * 86_400  # sessions span last n days

    for s_idx in range(n_sessions):
        t_session = t_base + s_idx * 86_400 + int(rng.integers(0, 3_600))
        rng.shuffle(users)  # random user order per session

        for user in users:
            # Sample a random slate of products for this user/session
            slate = [products[i] for i in rng.choice(len(products), size=min(slate_size, len(products)), replace=False)]

            for product in slate:
                logit      = _click_logit(user, product)
                # Add per-impression noise to simulate exploration/position bias
                noise      = float(rng.normal(0, 0.4))
                click_prob = _sigmoid(logit + noise)
                clicked    = int(rng.random() < click_prob)
                purchased  = int(clicked and rng.random() < purchase_rate)

                rows.append({
                    "user_id":    user.user_id,
                    "product_id": product.id,
                    "session_id": f"s_{s_idx:04d}_{user.user_id}",
                    "clicked":    clicked,
                    "purchased":  purchased,
                    "timestamp":  t_session + int(rng.integers(0, 600)),
                })

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    print(
        f"Generated {len(df):,} impressions | "
        f"CTR={df['clicked'].mean():.3f} | "
        f"CVR={df['purchased'].mean():.3f}"
    )
    return df


# ── Persistence helpers ───────────────────────────────────────────────────────

def save_users(users: List[UserProfile], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([u.to_dict() for u in users], f, indent=2)


def load_users(path: str | Path) -> List[UserProfile]:
    with open(path) as f:
        return [UserProfile.from_dict(d) for d in json.load(f)]


def build_dataset(
    n_users: int = N_USERS,
    n_sessions: int = N_SESSIONS,
    catalogue_seed: int = 42,
    interaction_seed: int = 0,
) -> Tuple[List[UserProfile], List[InsuranceProduct], pd.DataFrame]:
    """
    Convenience function: generate everything from scratch.
    Returns (users, products, interactions_df).
    """
    from .catalogue import generate_catalogue
    rng      = np.random.default_rng(catalogue_seed)
    products = generate_catalogue(seed=catalogue_seed)
    users    = _generate_users(n_users, rng)
    df       = generate_interactions(users, products, n_sessions=n_sessions, seed=interaction_seed)
    return users, products, df
