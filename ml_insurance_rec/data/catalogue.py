"""
Product catalogue — defines the insurance product schema and generates
a synthetic catalogue of 96 products with structured feature vectors.

The feature vector produced by InsuranceProduct.to_feature_vector() is
consumed by:
  • The Item Tower during two-tower training.
  • The FAISS index builder (items are embedded offline).
  • The ranking model (item features are part of every (user, item) row).

Feature vector layout — 50 dims total:
  [0:8]   type one-hot         8  (basic/comprehensive/adventure/…)
  [8]     price_per_day         1  (/ 20.0)
  [9]     coverage log-norm     1  (log1p / log1p(1e6))
  [10]    max_trip_days         1  (/ 90.0)
  [11]    rating                1  (/ 5.0)
  [12]    review_count log      1  (log1p / log1p(5000))
  [13]    freshness_score       1
  [14]    provider_tier         1  (/ 5.0)
  [15]    claim_score           1
  [16]    is_sponsored          1
  [17:33] feature flags        16  (binary)
  [33:42] destination flags     9  (binary)
  [42:50] style flags           8  (binary)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

# ── Vocabulary tables ──────────────────────────────────────────────────────────

TYPES = [
    "basic", "comprehensive", "adventure", "medical",
    "cfar", "family", "senior", "student",
]

PROVIDERS = [
    {"name": "TravelGuard",     "tier": 4, "claim_score": 0.88},
    {"name": "Allianz Travel",  "tier": 5, "claim_score": 0.92},
    {"name": "World Nomads",    "tier": 4, "claim_score": 0.87},
    {"name": "AXA Assistance",  "tier": 4, "claim_score": 0.85},
    {"name": "Seven Corners",   "tier": 3, "claim_score": 0.82},
    {"name": "Generali Global", "tier": 3, "claim_score": 0.80},
    {"name": "IMG Global",      "tier": 4, "claim_score": 0.86},
    {"name": "SafeTrip",        "tier": 2, "claim_score": 0.78},
]

# 16 binary insurance features (indexed 0-15)
FEATURES = [
    "Trip Cancellation",       # 0
    "Medical Emergency",       # 1
    "Evacuation",              # 2
    "Baggage Loss",            # 3
    "Travel Delay",            # 4
    "Adventure Sports",        # 5
    "Pre-existing Conditions", # 6
    "Cancel For Any Reason",   # 7
    "Rental Car",              # 8
    "Equipment Loss",          # 9
    "Search & Rescue",         # 10
    "Medical Repatriation",    # 11
    "24/7 Assistance",         # 12
    "Child Medical",           # 13
    "Pet Care Coverage",       # 14
    "Electronic Equipment",    # 15
]

# 9 destination types (indexed 0-8)
DESTINATIONS = [
    "domestic",       # 0
    "international",  # 1
    "schengen",       # 2
    "asia",           # 3
    "latam",          # 4
    "africa",         # 5
    "oceania",        # 6
    "adventure",      # 7
    "cruise",         # 8
]

# 8 travel styles (indexed 0-7)
STYLES = [
    "solo",       # 0
    "couple",     # 1
    "family",     # 2
    "group",      # 3
    "business",   # 4
    "senior",     # 5
    "student",    # 6
    "backpacker", # 7
]

# Which feature indices each product type covers
FEATURES_BY_TYPE: dict[str, List[int]] = {
    "basic":         [0, 3, 4, 12],
    "comprehensive": [0, 1, 2, 3, 4, 7, 6, 8],
    "adventure":     [1, 2, 5, 0, 9, 10, 11],
    "medical":       [1, 2, 6, 11, 12],
    "cfar":          [7, 0, 3, 4, 15],
    "family":        [0, 13, 2, 3, 4, 14],
    "senior":        [1, 6, 2, 0, 11, 12],
    "student":       [0, 1, 3, 4, 15],
}

DESTINATIONS_BY_TYPE: dict[str, List[int]] = {
    "basic":         [0, 1],
    "comprehensive": [0, 1, 2, 3, 4, 5, 6],
    "adventure":     [1, 7, 5, 3],
    "medical":       [1, 2, 3, 4, 5],
    "cfar":          [0, 1, 2, 8],
    "family":        [0, 1, 8, 3],
    "senior":        [0, 1, 8, 2],
    "student":       [1, 3, 4, 7],
}

STYLES_BY_TYPE: dict[str, List[int]] = {
    "basic":         [0, 1],
    "comprehensive": [4, 1, 3],
    "adventure":     [0, 7, 3],
    "medical":       [5, 4, 1],
    "cfar":          [4, 1, 3],
    "family":        [2, 1],
    "senior":        [5, 1],
    "student":       [6, 7, 0],
}

PRICE_RANGE: dict[str, tuple[float, float]] = {
    "basic":         (1.5,  4.0),
    "comprehensive": (5.0, 15.0),
    "adventure":     (6.0, 18.0),
    "medical":       (3.0,  9.0),
    "cfar":          (8.0, 20.0),
    "family":        (4.0, 12.0),
    "senior":        (6.0, 16.0),
    "student":       (1.0,  3.5),
}

COVERAGE_RANGE: dict[str, tuple[float, float]] = {
    "basic":         (15_000,    50_000),
    "comprehensive": (50_000,   500_000),
    "adventure":     (50_000,   300_000),
    "medical":       (100_000, 1_000_000),
    "cfar":          (20_000,   150_000),
    "family":        (30_000,   200_000),
    "senior":        (50_000,   500_000),
    "student":       (10_000,    50_000),
}

ITEM_FEATURE_DIM = 50


# ── Dataclass ──────────────────────────────────────────────────────────────────

@dataclass
class InsuranceProduct:
    id: str
    name: str
    provider: str
    provider_tier: int
    claim_score: float
    type: str                      # one of TYPES
    price_per_day: float
    coverage_amount: float
    max_trip_days: int
    rating: float
    review_count: int
    freshness_score: float         # [0, 1] — 1 = recently updated
    is_sponsored: bool
    feature_indices: List[int]     # subset of range(16)
    destination_indices: List[int] # subset of range(9)
    style_indices: List[int]       # subset of range(8)
    # Cached feature vector (set after construction)
    _fv: np.ndarray = field(default=None, repr=False, compare=False)

    # ── Feature vector ────────────────────────────────────────────────────────

    def to_feature_vector(self) -> np.ndarray:
        """Return the cached 50-dim float32 feature vector."""
        if self._fv is not None:
            return self._fv
        v = np.zeros(ITEM_FEATURE_DIM, dtype=np.float32)
        # [0:8] type one-hot
        v[TYPES.index(self.type)] = 1.0
        # [8:17] scalars
        v[8]  = self.price_per_day / 20.0
        v[9]  = np.log1p(self.coverage_amount) / np.log1p(1_000_000)
        v[10] = min(self.max_trip_days, 90) / 90.0
        v[11] = self.rating / 5.0
        v[12] = np.log1p(self.review_count) / np.log1p(5_000)
        v[13] = self.freshness_score
        v[14] = self.provider_tier / 5.0
        v[15] = self.claim_score
        v[16] = float(self.is_sponsored)
        # [17:33] feature flags
        for fi in self.feature_indices:
            if 0 <= fi < 16:
                v[17 + fi] = 1.0
        # [33:42] destination flags
        for di in self.destination_indices:
            if 0 <= di < 9:
                v[33 + di] = 1.0
        # [42:50] style flags
        for si in self.style_indices:
            if 0 <= si < 8:
                v[42 + si] = 1.0
        self._fv = v
        return v

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "provider": self.provider,
            "provider_tier": self.provider_tier, "claim_score": self.claim_score,
            "type": self.type, "price_per_day": self.price_per_day,
            "coverage_amount": self.coverage_amount, "max_trip_days": self.max_trip_days,
            "rating": self.rating, "review_count": self.review_count,
            "freshness_score": self.freshness_score, "is_sponsored": self.is_sponsored,
            "feature_indices": self.feature_indices,
            "destination_indices": self.destination_indices,
            "style_indices": self.style_indices,
            "features": [FEATURES[i] for i in self.feature_indices],
            "destinations": [DESTINATIONS[i] for i in self.destination_indices],
            "styles": [STYLES[i] for i in self.style_indices],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "InsuranceProduct":
        return cls(
            id=d["id"], name=d["name"], provider=d["provider"],
            provider_tier=d["provider_tier"], claim_score=d["claim_score"],
            type=d["type"], price_per_day=d["price_per_day"],
            coverage_amount=d["coverage_amount"], max_trip_days=d["max_trip_days"],
            rating=d["rating"], review_count=d["review_count"],
            freshness_score=d["freshness_score"], is_sponsored=d["is_sponsored"],
            feature_indices=d["feature_indices"],
            destination_indices=d["destination_indices"],
            style_indices=d["style_indices"],
        )


# ── Catalogue generator ────────────────────────────────────────────────────────

def generate_catalogue(n: int = 96, seed: int = 42) -> List[InsuranceProduct]:
    """
    Generate a deterministic synthetic catalogue of n insurance products.
    Products cycle through all 8 types with seeded-random attributes.
    """
    rng = np.random.default_rng(seed)
    products: List[InsuranceProduct] = []

    # Product name fragments per type
    name_prefix = {
        "basic":         ["EssentialCover", "TripSafe Lite", "QuickShield", "JourneyBasic"],
        "comprehensive": ["AllRisk Pro",    "TotalCover Plus","Premier Guard","GlobalElite"],
        "adventure":     ["AdventureElite","Summit Shield",  "WildTrack",   "ExpeditionPro"],
        "medical":       ["MediTravel Pro", "Emergency Plus", "HealthGuard", "MedEvac Shield"],
        "cfar":          ["Cancel Anytime", "FlexCancel",     "AnyReason",   "FreedomCancel"],
        "family":        ["FamilyFirst",    "KidsIncluded",   "FamilyJourney","HomeAway Pro"],
        "senior":        ["SeniorCare Pro", "GoldenTrip",     "ElderGuard",  "GoldYears"],
        "student":       ["BackpackerBasic","StudentExplorer","BudgetWanderer","GapYear Shield"],
    }

    for i in range(n):
        ptype    = TYPES[i % len(TYPES)]
        provider = PROVIDERS[int(rng.integers(len(PROVIDERS)))]
        pmin, pmax = PRICE_RANGE[ptype]
        cmin, cmax = COVERAGE_RANGE[ptype]

        name_list = name_prefix[ptype]
        name = f"{name_list[i % len(name_list)]} {provider['name'].split()[0]}"

        prod = InsuranceProduct(
            id               = f"ins_{i+1:04d}",
            name             = name,
            provider         = provider["name"],
            provider_tier    = provider["tier"],
            claim_score      = provider["claim_score"],
            type             = ptype,
            price_per_day    = round(float(rng.uniform(pmin, pmax)), 2),
            coverage_amount  = round(float(rng.uniform(cmin, cmax)) / 1_000) * 1_000,
            max_trip_days    = int(rng.choice([7, 14, 21, 30, 45, 60, 90])),
            rating           = round(float(rng.uniform(3.5, 5.0)), 1),
            review_count     = int(rng.integers(100, 5_000)),
            freshness_score  = round(float(rng.uniform(0.4, 1.0)), 2),
            is_sponsored     = bool(i < 6 and rng.random() > 0.65),
            feature_indices  = FEATURES_BY_TYPE[ptype],
            destination_indices = DESTINATIONS_BY_TYPE[ptype],
            style_indices    = STYLES_BY_TYPE[ptype],
        )
        products.append(prod)

    return products


def save_catalogue(products: List[InsuranceProduct], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([p.to_dict() for p in products], f, indent=2)


def load_catalogue(path: str | Path) -> List[InsuranceProduct]:
    with open(path) as f:
        return [InsuranceProduct.from_dict(d) for d in json.load(f)]
