"""Offline evaluation: AUC, NCE, and replay top-K match rate."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .features import FeatureExtractor
from .losses import normalized_cross_entropy
from .schema import Event
from .store import Store


@torch.no_grad()
def predict_proba(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    batch_size: int = 4096,
    logit_bias: float = 0.0,
) -> np.ndarray:
    """Sigmoid scores. ``logit_bias`` corrects for negative downsampling — see
    Bottou et al. (2013); odds_true = keep_fraction * odds_train, so we add
    log(keep_fraction) to the logit before the sigmoid.
    """
    model.eval()
    out = []
    for i in range(0, x.shape[0], batch_size):
        logits = model(x[i : i + batch_size]) + logit_bias
        out.append(torch.sigmoid(logits).cpu().numpy())
    if not out:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(out)


def auc(probs: np.ndarray, y: np.ndarray) -> float:
    if y.size == 0 or len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, probs))


def evaluate(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    background_ctr: float | None = None,
    logit_bias: float = 0.0,
) -> dict[str, float]:
    probs = predict_proba(model, x, logit_bias=logit_bias)
    y_np = y.cpu().numpy()
    return {
        "auc": auc(probs, y_np),
        "nce": normalized_cross_entropy(probs, y_np, background_ctr=background_ctr),
        "n": int(y_np.size),
        "ctr": float(y_np.mean()) if y_np.size else 0.0,
    }


def replay_topk_match_rate(
    model: torch.nn.Module,
    events: Iterable[Event],
    *,
    extractor: FeatureExtractor,
    store: Store,
    k: int = 5,
    logit_bias: float = 0.0,
) -> float:
    """Group events into request-batches, re-rank by the model, count clicks
    landing in top-K as matches. Approximates online performance from logs.

    We bucket events that share a (user_id, ~minute) key into a synthetic
    "request" — the same user typically scans a feed in a short window.
    """
    by_request: dict[tuple[str, int], list[Event]] = defaultdict(list)
    for e in events:
        by_request[(e.user_id, int(e.timestamp // 60))].append(e)

    matches = 0
    total_clicks = 0
    for (_, _), batch in by_request.items():
        clicked = [e for e in batch if e.clicked]
        if not clicked or len(batch) < 2:
            # singletons don't tell us anything about ordering
            total_clicks += len(clicked)
            continue
        x, _ = extractor.transform_events(batch, store=store)
        if x.shape[0] == 0:
            continue
        probs = predict_proba(model, torch.from_numpy(x), logit_bias=logit_bias)
        order = np.argsort(-probs)
        topk_idx = set(order[:k].tolist())
        for i, e in enumerate(batch):
            if e.clicked:
                total_clicks += 1
                if i in topk_idx:
                    matches += 1
    return matches / total_clicks if total_clicks else 0.0
