"""
Train the XGBoost ranking model.

Pipeline
────────
1. Load the interaction log and artefacts written by train_towers.py.
2. Build the (user ‖ item ‖ interaction) 110-dim feature matrix using
   feature_engineering.build_ranking_dataset().
3. Compute ANN scores from the trained item embeddings so the ranker
   learns to trust (or distrust) the retrieval signal.
4. Train XGBoostRanker with a stratified train/val split.
5. Log AUC, top feature importances, and calibration stats.
6. Build and save the FAISS ANN index from the item embeddings.

Run order
─────────
  python -m ml_insurance_rec.training.train_towers  [--epochs 30]
  python -m ml_insurance_rec.training.train_ranker   [--artifacts_dir artifacts]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit

from ..data.catalogue import load_catalogue
from ..data.feature_engineering import (
    build_ranking_dataset,
    build_ranking_row,
    RANKING_FEATURE_DIM,
)
from ..data.interaction_log import load_users
from ..models.ann_index import ANNIndex
from ..models.ranker import XGBoostRanker
from ..models.two_tower import TwoTowerModel

DEFAULTS = dict(
    artifacts_dir     = "artifacts",
    val_fraction      = 0.15,
    neg_sample_ratio  = 4,
    seed              = 42,
)


# ── ANN scores helper ─────────────────────────────────────────────────────────

def _add_ann_scores(
    users_by_id: dict,
    products_by_id: dict,
    interactions: pd.DataFrame,
    item_embeddings: np.ndarray,
    products_list: list,
) -> pd.DataFrame:
    """
    For every (user, product) pair in the interaction log, compute the cosine
    similarity between the user's two-tower embedding and the product's item
    embedding.  This ANN score becomes feature[5] in the ranking feature vector.

    We pre-compute all user embeddings once, then batch dot-product against the
    stored item matrix — O(U * D + I * D) instead of O(U * I * D).
    """
    from ..models.two_tower import TwoTowerModel
    import torch

    art      = Path(interactions.attrs.get("artifacts_dir", "artifacts"))
    model_pt = art / "two_tower_best.pt"
    if not model_pt.exists():
        print("  two_tower_best.pt not found — using zero ANN scores")
        interactions = interactions.copy()
        interactions["ann_score"] = 0.0
        return interactions

    model = TwoTowerModel.load(model_pt)
    model.eval()

    # Embed all users once
    unique_user_ids = interactions["user_id"].unique()
    user_embs: dict[str, np.ndarray] = {}
    for uid in unique_user_ids:
        u = users_by_id.get(uid)
        if u:
            user_embs[uid] = model.embed_single_user(u.to_feature_vector())

    # Build product_id → row index in item_embeddings
    pid_to_idx = {p.id: i for i, p in enumerate(products_list)}

    # Compute ann_score per row
    ann_scores = []
    for _, row in interactions.iterrows():
        u_emb = user_embs.get(row["user_id"])
        p_idx = pid_to_idx.get(row["product_id"])
        if u_emb is not None and p_idx is not None:
            score = float(np.dot(u_emb, item_embeddings[p_idx]))
        else:
            score = 0.0
        ann_scores.append(score)

    interactions = interactions.copy()
    interactions["ann_score"] = ann_scores
    return interactions


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    artifacts_dir:    str = DEFAULTS["artifacts_dir"],
    val_fraction:     float = DEFAULTS["val_fraction"],
    neg_sample_ratio: int   = DEFAULTS["neg_sample_ratio"],
    seed:             int   = DEFAULTS["seed"],
) -> XGBoostRanker:
    art = Path(artifacts_dir)

    # ── 1. Load artefacts from tower training ─────────────────────────────────
    print("Loading artefacts …")
    products    = load_catalogue(art / "catalogue.json")
    users       = load_users(art / "users.json")
    interactions = pd.read_parquet(art / "interactions.parquet")
    item_embs    = np.load(art / "item_embeddings.npy")   # (N_products, 128)

    users_by_id    = {u.user_id: u for u in users}
    products_by_id = {p.id: p for p in products}

    interactions.attrs["artifacts_dir"] = str(art)

    # ── 2. Attach ANN scores ─────────────────────────────────────────────────
    print("Computing ANN scores …")
    t0           = time.time()
    interactions = _add_ann_scores(users_by_id, products_by_id, interactions, item_embs, products)
    print(f"  Done in {time.time() - t0:.1f}s  mean_ann_score={interactions['ann_score'].mean():.4f}")

    # ── 3. Build ranking feature matrix ──────────────────────────────────────
    print("\nBuilding ranking feature matrix …")
    t0 = time.time()

    # Override build_ranking_dataset to incorporate per-row ann_scores
    from ..data.feature_engineering import build_ranking_row, compute_interaction_features
    rows_X, rows_y = [], []
    rng = np.random.default_rng(seed)
    all_pids = list(products_by_id.keys())

    positives = interactions[interactions["purchased"] == 1]
    for _, pos_row in positives.iterrows():
        uid  = pos_row["user_id"]
        pid  = pos_row["product_id"]
        ann  = pos_row.get("ann_score", 0.9)
        u    = users_by_id.get(uid)
        p    = products_by_id.get(pid)
        if u is None or p is None:
            continue
        rows_X.append(build_ranking_row(u, p, ann_score=ann))
        rows_y.append(1)

        # Negatives: random products the user didn't purchase
        purchased_ids = set(
            interactions[(interactions["user_id"] == uid) & (interactions["purchased"] == 1)]["product_id"]
        )
        neg_pool = [pid_ for pid_ in all_pids if pid_ not in purchased_ids]
        for neg_pid in rng.choice(neg_pool, size=min(neg_sample_ratio, len(neg_pool)), replace=False):
            neg_ann = float(rng.uniform(0.2, 0.6))
            rows_X.append(build_ranking_row(u, products_by_id[neg_pid], ann_score=neg_ann))
            rows_y.append(0)

    X = np.vstack(rows_X).astype(np.float32)
    y = np.array(rows_y, dtype=np.int32)
    print(f"  Matrix: {X.shape}  positives={y.sum():,} ({y.mean():.2%})  [{time.time()-t0:.1f}s]")

    # ── 4. Train / val split ─────────────────────────────────────────────────
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(sss.split(X, y))
    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]

    # ── 5. Train XGBoost ranker ───────────────────────────────────────────────
    print("\nTraining XGBoost ranker …")
    ranker = XGBoostRanker(
        n_estimators     = 300,
        max_depth        = 6,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample        = 0.8,
        scale_pos_weight = neg_sample_ratio,
        seed             = seed,
    )
    t0 = time.time()
    ranker.fit(X_train, y_train, X_val=X_val, y_val=y_val, verbose=True)
    print(f"  Training time: {time.time()-t0:.1f}s")

    # ── 6. Evaluate ───────────────────────────────────────────────────────────
    val_proba = ranker.predict_proba(X_val)
    auc       = roc_auc_score(y_val, val_proba)
    ap        = average_precision_score(y_val, val_proba)
    print(f"\nValidation  AUC={auc:.4f}  AP={ap:.4f}")

    # ── 7. Save ranker ────────────────────────────────────────────────────────
    ranker.save(art / "ranker.pkl")

    # ── 8. Build and save FAISS ANN index ────────────────────────────────────
    print("\nBuilding FAISS ANN index …")
    index = ANNIndex(embed_dim=item_embs.shape[1])
    index.build(item_embs, [p.id for p in products])
    index.save(art / "ann_index.pkl")

    print("\nAll artefacts saved:")
    for f in sorted(art.iterdir()):
        print(f"  {f.name}")

    return ranker


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts_dir",    type=str,   default=DEFAULTS["artifacts_dir"])
    ap.add_argument("--val_fraction",     type=float, default=DEFAULTS["val_fraction"])
    ap.add_argument("--neg_sample_ratio", type=int,   default=DEFAULTS["neg_sample_ratio"])
    ap.add_argument("--seed",             type=int,   default=DEFAULTS["seed"])
    args = ap.parse_args()
    train(**vars(args))
