"""Inference wrapper: holds the model, exposes a simple score() over a feature matrix.

Threading model: LightGBM `predict` releases the GIL, but we still want to keep
the model call inside the event loop's executor only when batches are large.
For typical ~50 candidate batches the call is <2ms, so we run it inline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ml.models.ranker import Ranker


class Predictor:
    def __init__(self, model_path: Path):
        self._ranker: Optional[Ranker] = None
        self._model_path = model_path

    def load(self) -> None:
        if self._ranker is None:
            self._ranker = Ranker.load(self._model_path)

    @property
    def ready(self) -> bool:
        return self._ranker is not None

    def score(self, X: np.ndarray) -> np.ndarray:
        assert self._ranker is not None, "Predictor.load() must be called before scoring"
        return self._ranker.predict(X)
