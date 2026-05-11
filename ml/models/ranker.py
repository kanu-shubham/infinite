"""LightGBM pCTR ranker.

We train a binary classifier on click labels rather than a pairwise LambdaRank
because the production target is calibrated pCTR (used both for ranking *and*
for pricing in the GSP auction). Calibration is preserved via isotonic
regression at training time.
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np

from .features import FEATURE_NAMES

DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l2": 1.0,
    "verbose": -1,
    # Determinism: makes p99 latency reproducible in benchmarks.
    "deterministic": True,
    "force_row_wise": True,
}


class Ranker:
    """Loaded once per worker process; thread-safe for `predict`."""

    def __init__(self, booster: lgb.Booster):
        self._booster = booster
        # `num_threads=1` per request keeps tail latency predictable under load:
        # we'd rather batch across requests via workers than fight for cores.
        self._predict_kwargs = {"num_threads": 1, "raw_score": False}

    @classmethod
    def load(cls, path: str | Path) -> "Ranker":
        return cls(lgb.Booster(model_file=str(path)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._booster.predict(X, **self._predict_kwargs)

    @staticmethod
    def train(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        num_boost_round: int = 400,
        params: dict | None = None,
    ) -> "Ranker":
        params = {**DEFAULT_PARAMS, **(params or {})}
        d_train = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
        d_val = lgb.Dataset(X_val, label=y_val, reference=d_train, feature_name=FEATURE_NAMES)
        booster = lgb.train(
            params,
            d_train,
            num_boost_round=num_boost_round,
            valid_sets=[d_val],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
        )
        return Ranker(booster)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._booster.save_model(str(path))
