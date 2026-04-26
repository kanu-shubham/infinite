"""Load a checkpoint and rank candidate posts for a user."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .config import DEFAULT_CHECKPOINT, FeatureSpec
from .features import FeatureExtractor
from .models import build_model
from .schema import ScoredPost
from .store import Store


class Ranker:
    """Wraps a trained model + its feature extractor."""

    def __init__(
        self,
        model: torch.nn.Module,
        extractor: FeatureExtractor,
        *,
        metadata: Optional[dict] = None,
    ) -> None:
        self.model = model.eval()
        self.extractor = extractor
        self.metadata = metadata or {}
        self.logit_bias = float(self.metadata.get("logit_bias", 0.0))

    @classmethod
    def load(cls, path: Path = DEFAULT_CHECKPOINT) -> "Ranker":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        meta = ckpt["metadata"]
        kind = meta["model_kind"]
        spec = FeatureSpec()
        extractor = FeatureExtractor(spec)

        kwargs: dict = {}
        if kind == "mlp":
            cfg = meta.get("config", {})
            kwargs = {
                "hidden_dims": tuple(cfg.get("hidden_dims", (64, 32))),
                "dropout": float(cfg.get("dropout", 0.1)),
            }
        model = build_model(kind, extractor.dim, **kwargs)
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, extractor, metadata=meta)

    @torch.no_grad()
    def score(
        self,
        user_id: str,
        post_ids: list[str],
        *,
        store: Store,
        timestamp: Optional[float] = None,
    ) -> list[float]:
        user = store.get_user(user_id)
        if user is None:
            raise KeyError(f"unknown user: {user_id}")
        rows: list[np.ndarray] = []
        kept_posts: list[str] = []
        for pid in post_ids:
            post = store.get_post(pid)
            if post is None:
                continue
            rows.append(
                self.extractor.transform(user, post, store=store, timestamp=timestamp)
            )
            kept_posts.append(pid)
        if not rows:
            return []
        x = torch.from_numpy(np.stack(rows))
        probs = torch.sigmoid(self.model(x) + self.logit_bias).cpu().numpy().tolist()
        # Re-align to original post_ids order, scoring missing posts as 0.
        score_map = dict(zip(kept_posts, probs))
        return [float(score_map.get(pid, 0.0)) for pid in post_ids]

    def rank(
        self,
        user_id: str,
        *,
        store: Store,
        candidate_post_ids: Optional[list[str]] = None,
        k: int = 20,
        timestamp: Optional[float] = None,
    ) -> list[ScoredPost]:
        if candidate_post_ids is None:
            candidate_post_ids = [p.post_id for p in store.all_posts()]
        if not candidate_post_ids:
            return []
        scores = self.score(user_id, candidate_post_ids, store=store, timestamp=timestamp)
        scored = [
            ScoredPost(post_id=pid, score=s)
            for pid, s in zip(candidate_post_ids, scores)
        ]
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:k]
