"""Synthetic hotel-bookings dataset.

The app ships without a data warehouse, so the "ingest" stage of the pipeline
generates a deterministic booking table instead. The generator deliberately
bakes in the messiness a real pipeline has to survive: missing values, a
skewed lead-time distribution, a mildly imbalanced cancellation target and
categorical levels that appear only a handful of times.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import DATASET_ROWS, DATASET_SEED

HOTEL_TYPES = ["City Hotel", "Resort Hotel"]
MARKET_SEGMENTS = ["Direct", "Online TA", "Offline TA/TO", "Corporate", "Groups"]
CUSTOMER_TYPES = ["Transient", "Transient-Party", "Contract", "Group"]
DEPOSIT_TYPES = ["No Deposit", "Non Refund", "Refundable"]
MEAL_PLANS = ["BB", "HB", "FB", "SC"]
ROOM_TYPES = ["A", "B", "C", "D", "E", "F"]
REGIONS = ["Europe", "North America", "Asia", "South America", "Africa", "Oceania"]
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

CATEGORICAL_FEATURES = [
    "hotel",
    "market_segment",
    "customer_type",
    "deposit_type",
    "meal_plan",
    "room_type",
    "region",
    "arrival_month",
]

NUMERIC_FEATURES = [
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "previous_cancellations",
    "booking_changes",
    "days_in_waiting_list",
    "special_requests",
    "required_car_parking",
    "is_repeated_guest",
    "adr",
]

TARGETS = {
    "cancellation": {
        "column": "is_canceled",
        "task": "classification",
        "label": "Booking cancellation",
        "description": "Will this booking be cancelled before check-in?",
        "positive_label": "Cancelled",
        "negative_label": "Honoured",
    },
    "price": {
        "column": "adr",
        "task": "regression",
        "label": "Average daily rate",
        "description": "What nightly rate will this booking be sold at?",
        "unit": "USD",
    },
}

DEFAULT_TARGET = "cancellation"


@dataclass(frozen=True)
class FeatureSpec:
    """Everything the UI needs to render one input control."""

    name: str
    kind: str  # "numeric" | "categorical"
    label: str
    options: List[str] | None = None
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    default: float | str | None = None


def _humanize(name: str) -> str:
    return name.replace("_", " ").capitalize()


def _sample_categorical(rng: np.random.Generator, values: List[str], weights, size: int):
    return rng.choice(values, size=size, p=np.asarray(weights) / np.sum(weights))


def generate_bookings(rows: int = DATASET_ROWS, seed: int = DATASET_SEED) -> pd.DataFrame:
    """Build the raw booking table. Deterministic for a given (rows, seed)."""
    rng = np.random.default_rng(seed)

    hotel = _sample_categorical(rng, HOTEL_TYPES, [0.62, 0.38], rows)
    market_segment = _sample_categorical(
        rng, MARKET_SEGMENTS, [0.14, 0.47, 0.19, 0.13, 0.07], rows
    )
    customer_type = _sample_categorical(
        rng, CUSTOMER_TYPES, [0.72, 0.16, 0.08, 0.04], rows
    )
    deposit_type = _sample_categorical(rng, DEPOSIT_TYPES, [0.86, 0.12, 0.02], rows)
    meal_plan = _sample_categorical(rng, MEAL_PLANS, [0.62, 0.22, 0.05, 0.11], rows)
    room_type = _sample_categorical(
        rng, ROOM_TYPES, [0.55, 0.14, 0.11, 0.10, 0.07, 0.03], rows
    )
    region = _sample_categorical(
        rng, REGIONS, [0.48, 0.21, 0.14, 0.08, 0.05, 0.04], rows
    )
    arrival_month = rng.choice(MONTHS, size=rows)

    # Lead time is heavily right-skewed in booking data.
    lead_time = np.clip(rng.gamma(shape=1.8, scale=45.0, size=rows), 0, 600).round()
    weekend_nights = rng.poisson(0.9, rows)
    week_nights = rng.poisson(2.4, rows)
    adults = np.clip(rng.poisson(1.8, rows), 1, 6)
    children = np.clip(rng.poisson(0.35, rows), 0, 4).astype(float)
    previous_cancellations = (rng.random(rows) < 0.07) * rng.integers(1, 5, rows)
    booking_changes = (rng.random(rows) < 0.18) * rng.integers(1, 4, rows)
    days_in_waiting_list = (rng.random(rows) < 0.05) * rng.integers(1, 120, rows)
    special_requests = np.clip(rng.poisson(0.6, rows), 0, 5)
    required_car_parking = (rng.random(rows) < 0.12).astype(int)
    is_repeated_guest = (rng.random(rows) < 0.06).astype(int)

    total_nights = np.maximum(weekend_nights + week_nights, 1)

    # ── Average daily rate: a signal the regression task can actually learn ──
    hotel_premium = np.where(hotel == "Resort Hotel", 34.0, 0.0)
    room_premium = pd.Series(room_type).map(
        {"A": 0.0, "B": 12.0, "C": 26.0, "D": 41.0, "E": 63.0, "F": 95.0}
    ).to_numpy()
    meal_premium = pd.Series(meal_plan).map(
        {"SC": 0.0, "BB": 9.0, "HB": 24.0, "FB": 38.0}
    ).to_numpy()
    season = pd.Series(arrival_month).map(
        {
            "January": -18.0, "February": -14.0, "March": -6.0, "April": 4.0,
            "May": 12.0, "June": 26.0, "July": 41.0, "August": 44.0,
            "September": 18.0, "October": 3.0, "November": -11.0, "December": 6.0,
        }
    ).to_numpy()

    adr = (
        72.0
        + hotel_premium
        + room_premium
        + meal_premium
        + season
        + 15.5 * adults
        + 9.0 * np.nan_to_num(children)
        - 0.035 * lead_time
        + 4.5 * special_requests
        + rng.normal(0, 13.0, rows)
    )
    adr = np.clip(adr, 25.0, None).round(2)

    # ── Cancellation: logistic model over the same drivers ──────────────────
    # Coefficients are sized so the target lands near a 35% positive rate —
    # roughly what public hotel-booking extracts show — and so a decent model
    # can separate the classes without the problem being trivial.
    logit = (
        -1.15
        + 0.0090 * lead_time
        + 1.60 * (deposit_type == "Non Refund")
        + 0.70 * (market_segment == "Groups")
        + 0.30 * (market_segment == "Online TA")
        - 0.90 * (market_segment == "Direct")
        - 0.40 * (customer_type == "Contract")
        + 1.20 * (previous_cancellations > 0)
        - 0.55 * special_requests
        - 0.65 * required_car_parking
        - 1.00 * is_repeated_guest
        + 0.012 * days_in_waiting_list
        - 0.20 * booking_changes
        + 0.0050 * (adr - adr.mean())
        + rng.normal(0, 0.30, rows)
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    is_canceled = (rng.random(rows) < probability).astype(int)

    frame = pd.DataFrame(
        {
            "booking_id": [f"BK-{i:06d}" for i in range(1, rows + 1)],
            "hotel": hotel,
            "market_segment": market_segment,
            "customer_type": customer_type,
            "deposit_type": deposit_type,
            "meal_plan": meal_plan,
            "room_type": room_type,
            "region": region,
            "arrival_month": arrival_month,
            "lead_time": lead_time.astype(int),
            "stays_in_weekend_nights": weekend_nights,
            "stays_in_week_nights": week_nights,
            "adults": adults,
            "children": children,
            "previous_cancellations": previous_cancellations,
            "booking_changes": booking_changes,
            "days_in_waiting_list": days_in_waiting_list,
            "special_requests": special_requests,
            "required_car_parking": required_car_parking,
            "is_repeated_guest": is_repeated_guest,
            "adr": adr,
            "is_canceled": is_canceled,
        }
    )

    # Real extracts have holes — the imputers in the pipeline exist for these.
    frame.loc[rng.random(rows) < 0.04, "children"] = np.nan
    frame.loc[rng.random(rows) < 0.02, "region"] = None
    frame.loc[rng.random(rows) < 0.015, "meal_plan"] = None

    return frame


@lru_cache(maxsize=4)
def load_bookings(rows: int = DATASET_ROWS, seed: int = DATASET_SEED) -> pd.DataFrame:
    """Cached dataset load — the ingest stage of every run reads through this."""
    return generate_bookings(rows, seed)


def feature_columns(target_key: str) -> Dict[str, List[str]]:
    """Split features by dtype, dropping the column being predicted."""
    target_column = TARGETS[target_key]["column"]
    numeric = [c for c in NUMERIC_FEATURES if c != target_column]
    categorical = [c for c in CATEGORICAL_FEATURES if c != target_column]
    return {"numeric": numeric, "categorical": categorical}


def feature_specs(target_key: str) -> List[FeatureSpec]:
    """Describe every input feature so the frontend can build a form for it."""
    frame = load_bookings()
    columns = feature_columns(target_key)
    specs: List[FeatureSpec] = []

    for name in columns["categorical"]:
        options = sorted(frame[name].dropna().unique().tolist())
        mode = frame[name].mode(dropna=True)
        specs.append(
            FeatureSpec(
                name=name,
                kind="categorical",
                label=_humanize(name),
                options=options,
                default=mode.iloc[0] if not mode.empty else options[0],
            )
        )

    for name in columns["numeric"]:
        series = frame[name].dropna()
        is_integral = float(series.round().sub(series).abs().max()) == 0.0
        specs.append(
            FeatureSpec(
                name=name,
                kind="numeric",
                label=_humanize(name),
                minimum=float(series.min()),
                maximum=float(series.max()),
                step=1.0 if is_integral else 0.01,
                default=round(float(series.median()), 2),
            )
        )

    return specs


def dataset_profile(target_key: str = DEFAULT_TARGET) -> dict:
    """Row counts, missing-value tallies and target distribution for the UI."""
    frame = load_bookings()
    target = TARGETS[target_key]
    column = target["column"]
    columns = feature_columns(target_key)
    used = columns["numeric"] + columns["categorical"]

    missing = {
        name: int(frame[name].isna().sum())
        for name in used
        if int(frame[name].isna().sum()) > 0
    }

    if target["task"] == "classification":
        counts = frame[column].value_counts().to_dict()
        distribution = {
            "kind": "classes",
            "classes": [
                {
                    "value": int(value),
                    "label": target["positive_label"] if value == 1 else target["negative_label"],
                    "count": int(count),
                    "share": round(count / len(frame), 4),
                }
                for value, count in sorted(counts.items())
            ],
        }
    else:
        series = frame[column].dropna()
        distribution = {
            "kind": "numeric",
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),
            "mean": round(float(series.mean()), 2),
            "median": round(float(series.median()), 2),
            "std": round(float(series.std()), 2),
        }

    return {
        "name": "hotel_bookings_synthetic",
        "rows": int(len(frame)),
        "feature_count": len(used),
        "numeric_features": columns["numeric"],
        "categorical_features": columns["categorical"],
        "missing_values": missing,
        "target": {
            "key": target_key,
            "column": column,
            "task": target["task"],
            "label": target["label"],
            "description": target["description"],
        },
        "target_distribution": distribution,
    }


def sample_rows(count: int = 5, seed: int | None = None) -> List[dict]:
    """A few raw bookings, used to preview the data and prefill the form."""
    frame = load_bookings()
    sampled = frame.sample(n=min(count, len(frame)), random_state=seed)
    return sampled.replace({np.nan: None}).to_dict(orient="records")
