"""Binary cross-entropy and Normalized Cross-Entropy.

NCE divides BCE by the entropy of the empirical click rate, making the loss
invariant to background CTR. A model that always predicts the prior gets
NCE = 1.0; lower is better.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


EPS = 1e-7


def bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets)


def _entropy(p: float) -> float:
    p = min(max(p, EPS), 1.0 - EPS)
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def normalized_cross_entropy(
    probs: torch.Tensor | "np.ndarray",  # type: ignore[name-defined]
    targets: torch.Tensor | "np.ndarray",  # type: ignore[name-defined]
    *,
    background_ctr: float | None = None,
) -> float:
    """Compute NCE = mean BCE / entropy(background CTR).

    Accepts either tensors or numpy arrays.
    """
    if hasattr(probs, "detach"):
        p = probs.detach().float().clamp(EPS, 1 - EPS).cpu().numpy()
    else:
        p = probs.clip(EPS, 1 - EPS)
    if hasattr(targets, "detach"):
        y = targets.detach().float().cpu().numpy()
    else:
        y = targets.astype("float32")

    if background_ctr is None:
        background_ctr = float(y.mean()) if y.size else 0.5

    # Lesson's formula uses y ∈ {-1, +1}; with y ∈ {0, 1} this is the
    # equivalent BCE-style numerator.
    nll = -(y * _safe_log(p) + (1 - y) * _safe_log(1 - p))
    denom = _entropy(background_ctr)
    return float(nll.mean() / denom) if denom > 0 else float("nan")


def _safe_log(x):
    import numpy as np

    return np.log(np.clip(x, EPS, 1.0))


class NCETorchLoss(torch.nn.Module):
    """Differentiable NCE for use as a training loss.

    The denominator is a constant w.r.t. the parameters, so this is just BCE
    rescaled. We expose it as its own loss so logged values are comparable to
    eval-time NCE.
    """

    def __init__(self, background_ctr: float):
        super().__init__()
        self.denom = _entropy(background_ctr)
        if self.denom <= 0:
            raise ValueError("background_ctr must be in (0, 1)")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return bce_loss(logits, targets) / self.denom
