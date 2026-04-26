"""Training loop with negative downsampling and temporal split."""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import DEFAULT_CHECKPOINT, FeatureSpec, TrainConfig
from .data import EventDataset, downsample_negatives, temporal_split
from .evaluate import evaluate, replay_topk_match_rate
from .features import FeatureExtractor
from .losses import NCETorchLoss
from .models import build_model
from .schema import TrainResponse
from .store import Store


def _seed_everything(seed: int) -> None:
    import random as _r

    _r.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(
    store: Store,
    cfg: Optional[TrainConfig] = None,
    *,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    extractor: Optional[FeatureExtractor] = None,
) -> TrainResponse:
    cfg = cfg or TrainConfig()
    _seed_everything(cfg.seed)
    extractor = extractor or FeatureExtractor(cfg.feature_spec)

    events = store.all_events()
    if not events:
        raise ValueError("store has no events to train on")

    train_events, val_events, test_events = temporal_split(
        events, val_fraction=cfg.val_fraction, test_fraction=cfg.test_fraction
    )
    if not train_events:
        raise ValueError("no training events after split")

    background_ctr = float(np.mean([1.0 if e.clicked else 0.0 for e in train_events]))
    if background_ctr <= 0 or background_ctr >= 1:
        raise ValueError(
            f"degenerate training labels (CTR={background_ctr:.4f}); need both classes"
        )

    train_events_ds = downsample_negatives(
        train_events, ratio=cfg.negative_downsample_ratio, seed=cfg.seed
    )

    n_neg_total = sum(1 for e in train_events if not e.clicked)
    n_pos_total = sum(1 for e in train_events if e.clicked)
    n_neg_kept = sum(1 for e in train_events_ds if not e.clicked)
    keep_fraction = n_neg_kept / n_neg_total if n_neg_total > 0 else 1.0
    # Calibrates predictions back to true CTR after negative downsampling.
    logit_bias = math.log(keep_fraction) if 0 < keep_fraction < 1 else 0.0

    train_set = EventDataset(train_events_ds, store=store, extractor=extractor)
    val_set = EventDataset(val_events, store=store, extractor=extractor)
    test_set = EventDataset(test_events, store=store, extractor=extractor)

    if len(train_set) == 0:
        raise ValueError("training dataset is empty after feature extraction")

    device = torch.device(cfg.device)
    model_kwargs: dict = {}
    if cfg.model_kind == "mlp":
        model_kwargs = {"hidden_dims": cfg.hidden_dims, "dropout": cfg.dropout}
    model = build_model(cfg.model_kind, extractor.dim, **model_kwargs).to(device)

    loss_fn = NCETorchLoss(background_ctr).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)

    last_train_loss = float("nan")
    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        n_batches = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            running += loss.item()
            n_batches += 1
        last_train_loss = running / max(n_batches, 1)

    val_metrics = evaluate(
        model, val_set.x.to(device), val_set.y.to(device), logit_bias=logit_bias
    )
    test_metrics = evaluate(
        model, test_set.x.to(device), test_set.y.to(device), logit_bias=logit_bias
    )
    replay = replay_topk_match_rate(
        model, test_events, extractor=extractor, store=store, k=5,
        logit_bias=logit_bias,
    )

    metadata = {
        "model_kind": cfg.model_kind,
        "feature_dim": extractor.dim,
        "background_ctr": background_ctr,
        "logit_bias": logit_bias,
        "negative_keep_fraction": keep_fraction,
        "n_pos_train": n_pos_total,
        "n_neg_train_total": n_neg_total,
        "n_neg_train_kept": n_neg_kept,
        "config": asdict(cfg),
        "metrics": {
            "train_loss": last_train_loss,
            "val": val_metrics,
            "test": test_metrics,
            "replay_topk_match_rate": replay,
        },
    }

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metadata": metadata,
        },
        checkpoint_path,
    )
    (checkpoint_path.with_suffix(".json")).write_text(json.dumps(metadata, indent=2, default=str))

    return TrainResponse(
        model_kind=cfg.model_kind,
        epochs=cfg.epochs,
        train_loss=last_train_loss,
        val_auc=val_metrics["auc"] if val_metrics["auc"] == val_metrics["auc"] else 0.0,
        val_nce=val_metrics["nce"],
        test_auc=test_metrics["auc"] if test_metrics["auc"] == test_metrics["auc"] else 0.0,
        test_nce=test_metrics["nce"],
        replay_topk_match_rate=replay,
        n_train=len(train_set),
        n_val=len(val_set),
        n_test=len(test_set),
    )
