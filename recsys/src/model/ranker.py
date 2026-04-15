"""
ranker.py
---------
XGBoost re-ranker — Stage 2 of the two-stage recommendation pipeline.

Pipeline recap
──────────────
Stage 1 — Retrieval (Two-Tower):
  Given a user, find top-K candidate hotels quickly using approximate
  nearest-neighbour search on the embedding index.
  Optimised for RECALL — cast a wide net.

Stage 2 — Ranking (XGBoost):
  Given the top-K candidates, score each (user, hotel) pair with rich
  features that were too expensive to compute for all hotels.
  Optimised for PRECISION — pick the best K' << K.

Why XGBoost for ranking?
  - Handles mixed feature types (numeric + categorical) natively
  - Works well with small datasets (K candidates per user)
  - Fast inference: scoring 50 candidates takes < 1ms
  - Easy to add new features without retraining the retrieval towers

In production this is called Learning-to-Rank (LTR).  More advanced
setups use LambdaMART (XGBoost has built-in support via objective='rank:ndcg').
We use regression here for simplicity but note the upgrade path.
"""
import logging

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_ranking_features(
    user_df: pd.DataFrame,
    hotel_df: pd.DataFrame,
    interactions: pd.DataFrame,
    user_encoder,
    hotel_encoder,
    tower_model,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Build the feature matrix for the XGBoost ranker.

    For each (user, hotel) interaction we concatenate:
      - Raw user features (encoded)
      - Raw hotel features (encoded)
      - The dot-product similarity score from the two towers
        (a single powerful feature summarising deep compatibility)

    This cross-feature between deep embeddings and hand-crafted features
    is what makes the two-stage approach so powerful.
    """
    import torch

    # Map IDs to row indices
    user_rows = user_df.iloc[interactions["user_id"].values]
    hotel_rows = hotel_df.iloc[interactions["hotel_id"].values]

    user_encoded = user_encoder.transform(user_rows.drop(columns=["user_id"]))
    hotel_encoded = hotel_encoder.transform(hotel_rows.drop(columns=["hotel_id"]))

    # Get tower similarity scores
    with torch.no_grad():
        u_tensor = torch.tensor(user_encoded, dtype=torch.float32)
        h_tensor = torch.tensor(hotel_encoded, dtype=torch.float32)
        tower_scores = tower_model(u_tensor, h_tensor).numpy()

    feature_df = pd.DataFrame(
        np.hstack([user_encoded, hotel_encoded]),
        columns=(
            [f"user_{c}" for c in user_encoder.cat_cols + user_encoder.num_cols]
            + [f"hotel_{c}" for c in hotel_encoder.cat_cols + hotel_encoder.num_cols]
        ),
    )
    feature_df["tower_similarity_score"] = tower_scores

    labels = interactions["rating"].values
    return feature_df, labels


class XGBoostRanker:
    """
    Thin wrapper around XGBRegressor for hotel ranking.

    Predicts a continuous relevance score for each (user, hotel) pair.
    Hotels are then sorted by this score and the top-K returned.
    """

    def __init__(self, config: dict):
        cfg = config["ranker"]
        self.model = XGBRegressor(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            random_state=cfg["random_state"],
            n_jobs=-1,
            verbosity=0,
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> dict:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        logger.info(f"Training XGBoost ranker on {len(X_train):,} pairs …")
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_val)

        # NDCG@10: standard ranking metric.
        # Measures whether the truly good hotels appear at the top of the list.
        # 1.0 = perfect ranking, 0.0 = random.
        ndcg = ndcg_score(
            y_val.reshape(1, -1),
            y_pred.reshape(1, -1),
            k=10,
        )
        logger.info(f"Ranker NDCG@10: {ndcg:.4f}")
        return {"ndcg_at_10": float(ndcg)}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def rank_candidates(
        self,
        candidate_hotel_ids: np.ndarray,
        user_encoded: np.ndarray,
        hotel_df: pd.DataFrame,
        hotel_encoder,
        tower_model,
        top_k: int = 10,
    ) -> list[int]:
        """
        Given a list of candidate hotel IDs (from the retrieval stage),
        score each one with the ranker and return the top-k hotel IDs.
        """
        import torch

        candidate_hotels = hotel_df[
            hotel_df["hotel_id"].isin(candidate_hotel_ids)
        ].reset_index(drop=True)

        hotel_encoded = hotel_encoder.transform(
            candidate_hotels.drop(columns=["hotel_id"])
        )

        # Repeat the user vector for each candidate
        n = len(candidate_hotels)
        user_tiled = np.tile(user_encoded, (n, 1))

        with torch.no_grad():
            u_tensor = torch.tensor(user_tiled, dtype=torch.float32)
            h_tensor = torch.tensor(hotel_encoded, dtype=torch.float32)
            tower_scores = tower_model(u_tensor, h_tensor).numpy()

        feature_df = pd.DataFrame(
            np.hstack([user_tiled, hotel_encoded]),
            columns=(
                [f"user_{c}" for c in hotel_encoder.cat_cols + hotel_encoder.num_cols]
                + [f"hotel_{c}" for c in hotel_encoder.cat_cols + hotel_encoder.num_cols]
            ),
        )
        feature_df["tower_similarity_score"] = tower_scores

        # Align column names with training features
        for col in self.model.get_booster().feature_names:
            if col not in feature_df.columns:
                feature_df[col] = 0.0
        feature_df = feature_df[self.model.get_booster().feature_names]

        scores = self.model.predict(feature_df)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return candidate_hotels["hotel_id"].iloc[top_indices].tolist()
