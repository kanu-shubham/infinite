"""Train a model from the persisted store and write a checkpoint."""
from __future__ import annotations

import argparse
import json

from ml.feed_ranking.config import DEFAULT_CHECKPOINT, DEFAULT_DATASET, TrainConfig
from ml.feed_ranking.store import Store
from ml.feed_ranking.train import train


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    p.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--model", choices=["mlp", "logreg"], default="mlp")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--negative-ratio", type=float, default=4.0)
    args = p.parse_args()

    store = Store.load_json(type(DEFAULT_DATASET)(args.dataset))
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_kind=args.model,
        negative_downsample_ratio=args.negative_ratio,
    )
    result = train(store, cfg, checkpoint_path=type(DEFAULT_CHECKPOINT)(args.checkpoint))
    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()
