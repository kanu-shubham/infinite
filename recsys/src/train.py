"""
train.py
--------
Full two-stage training pipeline:

  Step 1 — Fit feature encoders on user and hotel DataFrames
  Step 2 — Train Two-Tower retrieval model (PyTorch, BCE loss)
  Step 3 — Build hotel embedding index (pre-compute all hotel vectors)
  Step 4 — Build ranking features (raw features + tower similarity score)
  Step 5 — Train XGBoost re-ranker
  Step 6 — Evaluate (NDCG@10 quality gate)
  Step 7 — Log everything to MLflow; persist artefacts for the API

Run locally:
    cd recsys/
    make train
"""
import logging
import os
import sys

import joblib
import mlflow
import mlflow.pytorch
import numpy as np
import torch

from src.data_generator import generate_dataset, load_config
from src.model.towers import (
    EmbeddingIndex,
    FeatureEncoder,
    TwoTowerModel,
    train_two_tower,
)
from src.model.ranker import XGBoostRanker, build_ranking_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _resolve_mlflow_uri(cfg: dict) -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"])


def train(config_path: str = "config.yaml") -> dict:
    config = load_config(config_path)
    tower_cfg = config["towers"]
    ranker_cfg = config["ranker"]

    # ── MLflow ────────────────────────────────────────────────────────────────
    tracking_uri = _resolve_mlflow_uri(config)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # ── Data ──────────────────────────────────────────────────────────────────
    users, hotels, interactions = generate_dataset(config_path)
    logger.info(f"Users: {len(users):,}  Hotels: {len(hotels):,}  "
                f"Interactions: {len(interactions):,}")

    # ── Step 1: Feature encoders ───────────────────────────────────────────────
    user_encoder = FeatureEncoder()
    hotel_encoder = FeatureEncoder()

    user_feats_raw = users.drop(columns=["user_id"])
    hotel_feats_raw = hotels.drop(columns=["hotel_id"])

    user_encoded = user_encoder.fit_transform(user_feats_raw)
    hotel_encoded = hotel_encoder.fit_transform(hotel_feats_raw)

    logger.info(f"User feature dim: {user_encoder.output_dim}  "
                f"Hotel feature dim: {hotel_encoder.output_dim}")

    with mlflow.start_run() as run:
        # ── Step 2: Train Two-Tower retrieval model ───────────────────────────
        logger.info("── Stage 1: Training Two-Tower retrieval model ──────────")
        model = TwoTowerModel(
            user_input_dim=user_encoder.output_dim,
            hotel_input_dim=hotel_encoder.output_dim,
            hidden_dims=tower_cfg["hidden_dims"],
            embedding_dim=tower_cfg["embedding_dim"],
        )

        # Build training pairs: for each interaction use the encoded features
        inter_user_feats = user_encoded[interactions["user_id"].values]
        inter_hotel_feats = hotel_encoded[interactions["hotel_id"].values]
        labels = interactions["label"].values.astype(np.float32)

        epoch_losses = train_two_tower(
            model=model,
            user_feats=inter_user_feats,
            hotel_feats=inter_hotel_feats,
            labels=labels,
            epochs=tower_cfg["epochs"],
            batch_size=tower_cfg["batch_size"],
            lr=tower_cfg["learning_rate"],
        )

        mlflow.log_params({
            "embedding_dim": tower_cfg["embedding_dim"],
            "hidden_dims": str(tower_cfg["hidden_dims"]),
            "tower_epochs": tower_cfg["epochs"],
            "tower_lr": tower_cfg["learning_rate"],
            "n_users": len(users),
            "n_hotels": len(hotels),
            "n_interactions": len(interactions),
        })
        for epoch, loss in enumerate(epoch_losses, 1):
            mlflow.log_metric("tower_train_loss", loss, step=epoch)

        # ── Step 3: Build hotel embedding index ───────────────────────────────
        logger.info("── Building hotel embedding index ───────────────────────")
        model.eval()
        with torch.no_grad():
            all_hotel_embeddings = (
                model.get_hotel_embeddings(
                    torch.tensor(hotel_encoded, dtype=torch.float32)
                )
                .numpy()
            )
        embedding_index = EmbeddingIndex(
            hotel_embeddings=all_hotel_embeddings,
            hotel_ids=hotels["hotel_id"].values,
        )

        # ── Step 4 + 5: Build ranking features and train XGBoost ranker ──────
        logger.info("── Stage 2: Training XGBoost ranker ─────────────────────")
        ranking_features, rating_labels = build_ranking_features(
            user_df=users,
            hotel_df=hotels,
            interactions=interactions,
            user_encoder=user_encoder,
            hotel_encoder=hotel_encoder,
            tower_model=model,
        )

        ranker = XGBoostRanker(config)
        ranker_metrics = ranker.fit(ranking_features, rating_labels)

        mlflow.log_params({
            "ranker_n_estimators": ranker_cfg["n_estimators"],
            "ranker_max_depth": ranker_cfg["max_depth"],
            "ranker_lr": ranker_cfg["learning_rate"],
            "n_candidates": ranker_cfg["n_candidates"],
        })
        mlflow.log_metrics(ranker_metrics)
        mlflow.log_metric("tower_final_loss", epoch_losses[-1])

        # ── Step 6: Quality gate ──────────────────────────────────────────────
        ndcg = ranker_metrics["ndcg_at_10"]
        threshold = config["monitoring"]["min_ndcg_threshold"]
        logger.info(f"NDCG@10: {ndcg:.4f}  (threshold: {threshold})")

        # ── Step 7: Persist artefacts ─────────────────────────────────────────
        os.makedirs("models", exist_ok=True)

        torch.save(model.state_dict(), "models/two_tower.pt")
        joblib.dump(user_encoder, "models/user_encoder.pkl")
        joblib.dump(hotel_encoder, "models/hotel_encoder.pkl")
        joblib.dump(embedding_index, "models/embedding_index.pkl")
        joblib.dump(ranker, "models/ranker.pkl")

        # Save model architecture config for loading at inference
        joblib.dump(
            {
                "user_input_dim": user_encoder.output_dim,
                "hotel_input_dim": hotel_encoder.output_dim,
                "hidden_dims": tower_cfg["hidden_dims"],
                "embedding_dim": tower_cfg["embedding_dim"],
            },
            "models/tower_config.pkl",
        )

        # Log all artefacts to MLflow
        mlflow.log_artifacts("models/", artifact_path="model")

        logger.info(f"All artefacts saved. MLflow run: {run.info.run_id}")

    logger.info("Training complete.")
    return {**ranker_metrics, "tower_final_loss": epoch_losses[-1]}


if __name__ == "__main__":
    metrics = train()
    config = load_config()
    threshold = config["monitoring"]["min_ndcg_threshold"]
    if metrics["ndcg_at_10"] < threshold:
        logger.error(f"QUALITY GATE FAILED — NDCG@10 {metrics['ndcg_at_10']:.4f} < {threshold}")
        sys.exit(1)
    logger.info("Quality gate passed.")
