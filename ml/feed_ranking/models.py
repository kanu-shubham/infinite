"""Models: sparse linear baseline + MLP."""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    """Linear model with a sigmoid head. Cheap and Spark-friendly in spirit."""

    kind = "logreg"

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)  # logits


class MLPClassifier(nn.Module):
    """Fully-connected DNN with ReLU + dropout + sigmoid output (logit form)."""

    kind = "mlp"

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int] = (64, 32),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # logits


def build_model(kind: str, input_dim: int, **kwargs) -> nn.Module:
    if kind == "logreg":
        return LogisticRegression(input_dim)
    if kind == "mlp":
        return MLPClassifier(input_dim, **kwargs)
    raise ValueError(f"unknown model kind: {kind}")
