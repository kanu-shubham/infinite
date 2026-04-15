"""
evaluate.py
-----------
Quality gate for the recommendation system.

Loads trained artefacts and runs an end-to-end evaluation:
  1. For a sample of users, retrieve top-50 candidates via the embedding index
  2. Re-rank candidates with XGBoost
  3. Compute NDCG@10 against held-out ground truth ratings

Exits with code 1 if NDCG@10 falls below the configured threshold —
this fails the CI pipeline and blocks the Docker build.

Run locally:
    cd recsys/
    make evaluate
"""
import logging
import sys

import joblib
import numpy as np
import torch
from sklearn.metrics import ndcg_score

from src.data_generator import generate_dataset, load_config
from src.model.towers import TwoTowerModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def evaluate(config_path: str = "config.yaml") -> dict:
    config = load_config(config_path)
    threshold = config["monitoring"]["min_ndcg_threshold"]
    top_k = config["serving"]["top_k"]
    n_candidates = config["ranker"]["n_candidates"]

    logger.info("Loading trained artefacts …")
    user_encoder = joblib.load("models/user_encoder.pkl")
    hotel_encoder = joblib.load("models/hotel_encoder.pkl")
    embedding_index = joblib.load("models/embedding_index.pkl")
    ranker = joblib.load("models/ranker.pkl")
    tower_config = joblib.load("models/tower_config.pkl")

    tower_model = TwoTowerModel(**tower_config)
    tower_model.load_state_dict(torch.load("models/two_tower.pt", map_location="cpu"))
    tower_model.eval()

    # Use a fresh random seed — data not seen during training
    users, hotels, interactions = generate_dataset(config_path)
    rng = np.random.RandomState(999)
    sample_user_ids = rng.choice(users["user_id"].values, size=200, replace=False)

    ndcg_scores = []

    for uid in sample_user_ids:
        # Ground-truth: ratings this user gave to all hotels they interacted with
        user_ints = interactions[interactions["user_id"] == uid]
        if len(user_ints) < top_k:
            continue

        # Encode the user
        user_row = users[users["user_id"] == uid].drop(columns=["user_id"])
        user_enc = user_encoder.transform(user_row)

        with torch.no_grad():
            user_emb = (
                tower_model.get_user_embedding(
                    torch.tensor(user_enc, dtype=torch.float32)
                )
                .numpy()
                .squeeze()
            )

        # Stage 1: retrieve candidates
        candidate_ids = embedding_index.get_top_k(user_emb, k=n_candidates)

        # Stage 2: rank candidates
        ranked_ids = ranker.rank_candidates(
            candidate_hotel_ids=candidate_ids,
            user_encoded=user_enc,
            hotel_df=hotels,
            hotel_encoder=hotel_encoder,
            tower_model=tower_model,
            top_k=top_k,
        )

        # Compare ranked list against ground truth ratings
        truth_ratings = dict(zip(user_ints["hotel_id"], user_ints["rating"]))
        true_relevance = np.array(
            [truth_ratings.get(hid, 0.0) for hid in ranked_ids]
        )
        # Score of 1 for each position (ranked order is the prediction)
        predicted_scores = np.arange(top_k, 0, -1, dtype=float)

        if true_relevance.sum() > 0:
            score = ndcg_score(
                true_relevance.reshape(1, -1),
                predicted_scores.reshape(1, -1),
            )
            ndcg_scores.append(score)

    mean_ndcg = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
    metrics = {"ndcg_at_10": round(mean_ndcg, 4), "n_users_evaluated": len(ndcg_scores)}

    logger.info("── Evaluation Results ────────────────────────────────")
    logger.info(f"  NDCG@10           : {mean_ndcg:.4f}  (threshold: {threshold})")
    logger.info(f"  Users evaluated   : {len(ndcg_scores)}")
    logger.info("──────────────────────────────────────────────────────")

    if mean_ndcg < threshold:
        logger.error(
            f"QUALITY GATE FAILED — NDCG@10 {mean_ndcg:.4f} < {threshold}. "
            "Model will NOT be deployed."
        )
        sys.exit(1)

    logger.info("QUALITY GATE PASSED — model is ready for deployment.")
    return metrics


if __name__ == "__main__":
    evaluate()
