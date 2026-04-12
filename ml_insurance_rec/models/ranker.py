"""
Ranking model — XGBoost binary classifier trained to predict P(purchase).

Inputs: 110-dim (user ‖ item ‖ interaction) feature vector per candidate.
Output: purchase probability in [0, 1].

Why XGBoost here instead of a DNN?
  • Tabular features with mixed scales, binary flags, and ratio features
    are where gradient-boosted trees consistently match or beat DNNs.
  • Fast inference (microseconds per candidate) with no GPU required.
  • Feature importance is directly readable — useful for debugging.
  • In practice, companies (Airbnb, Expedia, LinkedIn) often use GBDT
    for the final ranking stage even when they use DNNs in retrieval.

In a real system you would also sweep hyperparameters with Optuna/Ray and
evaluate with nDCG@K rather than AUC.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    xgb = None

from ..data.feature_engineering import RANKING_FEATURE_NAMES, RANKING_FEATURE_DIM


class XGBoostRanker:
    """
    Thin wrapper around XGBClassifier that adds feature-name tracking,
    feature importance logging, and consistent save/load helpers.

    Parameters
    ----------
    n_estimators  : number of boosting rounds.
    max_depth     : maximum tree depth.
    learning_rate : shrinkage factor (η).
    subsample     : fraction of training rows used per tree.
    colsample     : fraction of features used per tree.
    scale_pos_weight : ratio neg/pos to correct class imbalance.
    seed          : random seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators:     int   = 300,
        max_depth:        int   = 6,
        learning_rate:    float = 0.05,
        subsample:        float = 0.8,
        colsample:        float = 0.8,
        scale_pos_weight: float = 4.0,   # neg_sample_ratio from feature_engineering
        seed:             int   = 42,
    ):
        if not _XGB_AVAILABLE:
            raise ImportError("xgboost is not installed.  Run: pip install xgboost")

        self._params = dict(
            n_estimators     = n_estimators,
            max_depth        = max_depth,
            learning_rate    = learning_rate,
            subsample        = subsample,
            colsample_bytree = colsample,
            scale_pos_weight = scale_pos_weight,
            objective        = "binary:logistic",
            eval_metric      = "auc",
            random_state     = seed,
            n_jobs           = -1,
            verbosity        = 0,
        )
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: List[str] = RANKING_FEATURE_NAMES

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   Optional[np.ndarray] = None,
        y_val:   Optional[np.ndarray] = None,
        early_stopping_rounds: int = 30,
        verbose: bool = True,
    ) -> "XGBoostRanker":
        """
        Train the ranker on (X_train, y_train).

        X_train : float32 (N, 110) ranking feature matrix
        y_train : int32   (N,)     binary purchase labels

        If validation data is supplied, early stopping is applied.
        """
        assert X_train.shape[1] == RANKING_FEATURE_DIM, (
            f"Expected {RANKING_FEATURE_DIM} features, got {X_train.shape[1]}"
        )
        self.model = xgb.XGBClassifier(**self._params)

        fit_kwargs: dict = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
            fit_kwargs["verbose"] = verbose

        self.model.fit(X_train, y_train, **fit_kwargs)

        if verbose:
            self._print_top_features(n=15)

        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return P(purchase) for each row in X.

        X      : float32 (N, 110)
        returns: float32 (N,)
        """
        assert self.model is not None, "Model not trained.  Call fit() first."
        return self.model.predict_proba(X)[:, 1].astype(np.float32)

    def score_single(self, row: np.ndarray) -> float:
        """Score one (110,) feature row. Returns scalar purchase probability."""
        return float(self.predict_proba(row[None])[0])

    # ── Feature importance ────────────────────────────────────────────────────

    def _print_top_features(self, n: int = 15) -> None:
        importances = self.model.feature_importances_
        idx_sorted  = np.argsort(importances)[::-1][:n]
        print("\nTop feature importances (gain):")
        for rank, i in enumerate(idx_sorted, 1):
            name = self.feature_names[i] if i < len(self.feature_names) else f"feat_{i}"
            print(f"  {rank:>2}. {name:<35} {importances[i]:.4f}")

    def feature_importance_dict(self) -> dict[str, float]:
        if self.model is None:
            return {}
        importances = self.model.feature_importances_
        return {
            (self.feature_names[i] if i < len(self.feature_names) else f"feat_{i}"): float(importances[i])
            for i in range(len(importances))
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"params": self._params, "feature_names": self.feature_names}
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        # Save XGBoost model separately in its native JSON format for portability
        model_path = path.with_suffix(".xgb.json")
        self.model.save_model(str(model_path))
        print(f"XGBoostRanker saved → {path}, {model_path}")

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostRanker":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        obj = cls(**{k: v for k, v in payload["params"].items() if k not in ("n_jobs", "verbosity", "objective", "eval_metric")})
        obj.feature_names = payload["feature_names"]
        model_path = Path(path).with_suffix(".xgb.json")
        obj.model = xgb.XGBClassifier()
        obj.model.load_model(str(model_path))
        print(f"XGBoostRanker loaded ← {path}")
        return obj
