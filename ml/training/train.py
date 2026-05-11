"""Offline training entry point. Writes models/artifacts/ranker.txt."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from ml.models.ranker import Ranker
from ml.training.synthetic_data import generate

ARTIFACT_PATH = Path(__file__).resolve().parents[1] / "models" / "artifacts" / "ranker.txt"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=200_000)
    p.add_argument("--rounds", type=int, default=400)
    p.add_argument("--out", type=Path, default=ARTIFACT_PATH)
    args = p.parse_args()

    t0 = time.time()
    X, y = generate(args.rows)
    print(f"[data] {X.shape[0]} rows, base-rate={y.mean():.4f}, {time.time()-t0:.1f}s")

    split = int(0.85 * len(X))
    X_tr, X_va = X[:split], X[split:]
    y_tr, y_va = y[:split], y[split:]

    t0 = time.time()
    ranker = Ranker.train(X_tr, y_tr, X_va, y_va, num_boost_round=args.rounds)
    print(f"[train] {time.time()-t0:.1f}s")

    preds = ranker.predict(X_va)
    print(f"[val] logloss={log_loss(y_va, preds):.4f} auc={roc_auc_score(y_va, preds):.4f}")

    ranker.save(args.out)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
