"""Synthetic data generation + PyTorch Dataset.

We fabricate users, posts, and impressions with a *learnable* CTR signal so
that a model trained on this data exhibits realistic gains over random.

The latent click probability for (user, post) is built from:
  - is_connected to author (strong +)
  - cosine-ish topic affinity (medium +)
  - post recency, decayed (medium +)
  - engagement counts, log-scaled (medium +)
  - small per-user bias

A logistic on these signals is sampled to produce binary labels.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import POST_TYPES, TOPICS
from .features import FeatureExtractor
from .schema import Event, Post, User
from .store import Store


# ----- generation -----------------------------------------------------------

@dataclass
class GenerationConfig:
    n_users: int = 500
    n_posts: int = 2000
    n_impressions: int = 50_000
    avg_connections: int = 30
    seed: int = 17
    span_days: int = 14


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def generate_dataset(cfg: GenerationConfig, *, store: Store | None = None) -> Store:
    rng = random.Random(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)
    store = store or Store()
    now = time.time()

    # Users
    for i in range(cfg.n_users):
        affinities = {
            t: float(np_rng.beta(0.5, 5))  # mostly small, occasional spikes
            for t in TOPICS
        }
        user = User(
            user_id=f"u_{i}",
            created_at=now - rng.randint(30, 365 * 3) * 86400,
            num_connections=int(np_rng.poisson(cfg.avg_connections)),
            num_followers=int(np_rng.poisson(cfg.avg_connections * 2)),
            topic_affinities=affinities,
        )
        store.upsert_user(user)

    user_ids = [u.user_id for u in store.all_users()]

    # Connections — sparse random graph
    for uid in user_ids:
        targets = rng.sample(user_ids, k=min(rng.randint(5, cfg.avg_connections), len(user_ids)))
        for t in targets:
            if t != uid:
                store.connect(uid, t)

    # Posts
    span_seconds = cfg.span_days * 86400
    for j in range(cfg.n_posts):
        author = rng.choice(user_ids)
        created = now - rng.uniform(0, span_seconds)
        post = Post(
            post_id=f"p_{j}",
            author_id=author,
            created_at=created,
            post_type=rng.choice(POST_TYPES),
            topic=rng.choice(TOPICS),
            text_length=int(np_rng.gamma(2.0, 80.0)),
            num_likes=int(np_rng.poisson(5)),
            num_comments=int(np_rng.poisson(1.5)),
            num_shares=int(np_rng.poisson(0.5)),
        )
        store.upsert_post(post)

    posts = store.all_posts()

    # Impressions: pick a user, sample candidate posts (recency-biased), serve
    # them, then label using the latent CTR.
    for _ in range(cfg.n_impressions):
        user = store.get_user(rng.choice(user_ids))
        post = rng.choice(posts)
        ts = max(post.created_at, now - rng.uniform(0, span_seconds))
        is_conn = store.is_connected(user.user_id, post.author_id)

        post_age_h = max(0.0, (ts - post.created_at) / 3600.0)
        recency = math.exp(-post_age_h / 48.0)            # decays over ~2 days
        affinity = user.topic_affinities.get(post.topic, 0.0)
        engagement = math.log1p(post.num_likes + 2 * post.num_comments + 3 * post.num_shares)

        logit = (
            -3.6                      # background CTR ~ 2.6%
            + 1.4 * (1.0 if is_conn else 0.0)
            + 2.5 * affinity
            + 0.9 * recency
            + 0.25 * engagement
            + np_rng.normal(0, 0.4)
        )
        clicked = np_rng.random() < _sigmoid(logit)
        store.add_event(
            Event(
                user_id=user.user_id,
                post_id=post.post_id,
                timestamp=ts,
                clicked=bool(clicked),
                is_connected=is_conn,
            )
        )

    return store


# ----- splits ---------------------------------------------------------------

def temporal_split(
    events: list[Event], *, val_fraction: float, test_fraction: float
) -> tuple[list[Event], list[Event], list[Event]]:
    """Sort by timestamp, then split into [train | val | test]."""
    if not events:
        return [], [], []
    sorted_events = sorted(events, key=lambda e: e.timestamp)
    n = len(sorted_events)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    n_train = n - n_val - n_test
    return (
        sorted_events[:n_train],
        sorted_events[n_train : n_train + n_val],
        sorted_events[n_train + n_val :],
    )


def downsample_negatives(
    events: list[Event], *, ratio: float, seed: int = 0
) -> list[Event]:
    """Keep all positives, sample N = ratio * positives negatives."""
    pos = [e for e in events if e.clicked]
    neg = [e for e in events if not e.clicked]
    if not pos or not neg:
        return list(events)
    target = min(len(neg), int(len(pos) * ratio))
    rng = random.Random(seed)
    sampled_neg = rng.sample(neg, target)
    out = pos + sampled_neg
    rng.shuffle(out)
    return out


# ----- torch dataset --------------------------------------------------------

class EventDataset(Dataset):
    """Materialises features once for fast iteration."""

    def __init__(
        self,
        events: Iterable[Event],
        *,
        store: Store,
        extractor: FeatureExtractor,
    ) -> None:
        x, y = extractor.transform_events(events, store=store)
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]
