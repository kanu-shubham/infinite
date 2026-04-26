"""(user, post, context) -> fixed-width feature vector.

Numeric features are log1p-normalised where they are heavy-tailed (counts,
text length). Categoricals are one-hot. The final layout is:

    [ numeric (12) | post_type one-hot (5) | topic one-hot (10) ]
"""
from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Iterable, Optional

import numpy as np

from .config import FeatureSpec
from .schema import Event, Post, User
from .store import Store


def _log1p(x: float) -> float:
    return math.log1p(max(x, 0.0))


def _topic_affinity(user: User, post: Post) -> float:
    return float(user.topic_affinities.get(post.topic, 0.0))


def _hour_of_day(ts: float) -> float:
    # Normalised to [0, 1).
    return datetime.fromtimestamp(ts, tz=timezone.utc).hour / 24.0


class FeatureExtractor:
    """Stateless feature builder. Vocabularies live on the FeatureSpec."""

    def __init__(self, spec: FeatureSpec | None = None) -> None:
        self.spec = spec or FeatureSpec()
        self._post_type_index = {t: i for i, t in enumerate(self.spec.post_type_vocab)}
        self._topic_index = {t: i for i, t in enumerate(self.spec.topic_vocab)}

    @property
    def dim(self) -> int:
        return self.spec.dim

    def transform(
        self,
        user: User,
        post: Post,
        *,
        store: Store,
        timestamp: Optional[float] = None,
        is_connected: Optional[bool] = None,
    ) -> np.ndarray:
        ts = timestamp if timestamp is not None else time.time()
        if is_connected is None:
            is_connected = store.is_connected(user.user_id, post.author_id)

        author = store.get_user(post.author_id)
        author_followers = author.num_followers if author else 0
        post_age_h = max(0.0, (ts - post.created_at) / 3600.0)
        user_age_d = max(0.0, (ts - user.created_at) / 86400.0)

        numeric = np.array(
            [
                _log1p(user_age_d),
                _log1p(user.num_connections),
                _log1p(user.num_followers),
                _log1p(post_age_h),
                _log1p(post.num_likes),
                _log1p(post.num_comments),
                _log1p(post.num_shares),
                _log1p(post.text_length),
                _log1p(author_followers),
                1.0 if is_connected else 0.0,
                _topic_affinity(user, post),
                _hour_of_day(ts),
            ],
            dtype=np.float32,
        )

        post_type = np.zeros(len(self.spec.post_type_vocab), dtype=np.float32)
        post_type[self._post_type_index[post.post_type]] = 1.0
        topic = np.zeros(len(self.spec.topic_vocab), dtype=np.float32)
        topic[self._topic_index[post.topic]] = 1.0

        return np.concatenate([numeric, post_type, topic])

    def transform_pairs(
        self,
        pairs: Iterable[tuple[User, Post]],
        *,
        store: Store,
        timestamp: Optional[float] = None,
    ) -> np.ndarray:
        rows = [self.transform(u, p, store=store, timestamp=timestamp) for u, p in pairs]
        if not rows:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack(rows)

    def transform_events(
        self,
        events: Iterable[Event],
        *,
        store: Store,
    ) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for e in events:
            user = store.get_user(e.user_id)
            post = store.get_post(e.post_id)
            if user is None or post is None:
                continue
            xs.append(
                self.transform(
                    user,
                    post,
                    store=store,
                    timestamp=e.timestamp,
                    is_connected=e.is_connected,
                )
            )
            ys.append(1.0 if e.clicked else 0.0)
        if not xs:
            return (
                np.zeros((0, self.dim), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )
        return np.stack(xs), np.array(ys, dtype=np.float32)
