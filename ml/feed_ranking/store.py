"""Tiny in-memory store. Persistable to JSON for the demo.

In production this would be backed by a relational DB for entities and a
log-structured store (Kafka + parquet) for events.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Iterable, Optional

from .schema import Event, Post, User


class Store:
    def __init__(self) -> None:
        self._users: dict[str, User] = {}
        self._posts: dict[str, Post] = {}
        self._events: list[Event] = []
        # adjacency: user_id -> set of connected user_ids
        self._connections: dict[str, set[str]] = {}
        self._lock = threading.RLock()

    # ---- entities ----------------------------------------------------------
    def upsert_user(self, user: User) -> User:
        with self._lock:
            self._users[user.user_id] = user
            self._connections.setdefault(user.user_id, set())
            return user

    def upsert_post(self, post: Post) -> Post:
        with self._lock:
            self._posts[post.post_id] = post
            return post

    def get_user(self, user_id: str) -> Optional[User]:
        return self._users.get(user_id)

    def get_post(self, post_id: str) -> Optional[Post]:
        return self._posts.get(post_id)

    def all_users(self) -> list[User]:
        return list(self._users.values())

    def all_posts(self) -> list[Post]:
        return list(self._posts.values())

    # ---- connections -------------------------------------------------------
    def connect(self, a: str, b: str) -> None:
        with self._lock:
            self._connections.setdefault(a, set()).add(b)
            self._connections.setdefault(b, set()).add(a)

    def is_connected(self, a: str, b: str) -> bool:
        return b in self._connections.get(a, set())

    # ---- events ------------------------------------------------------------
    def add_event(self, event: Event) -> Event:
        with self._lock:
            if event.is_connected is None:
                post = self._posts.get(event.post_id)
                event = event.model_copy(
                    update={
                        "is_connected": (
                            post is not None
                            and self.is_connected(event.user_id, post.author_id)
                        )
                    }
                )
            self._events.append(event)
            return event

    def add_events(self, events: Iterable[Event]) -> None:
        for e in events:
            self.add_event(e)

    def all_events(self) -> list[Event]:
        return list(self._events)

    # ---- persistence -------------------------------------------------------
    def save_json(self, path: Path) -> None:
        payload = {
            "users": [u.model_dump() for u in self._users.values()],
            "posts": [p.model_dump() for p in self._posts.values()],
            "events": [e.model_dump() for e in self._events],
            "connections": {k: sorted(v) for k, v in self._connections.items()},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))

    @classmethod
    def load_json(cls, path: Path) -> "Store":
        data = json.loads(path.read_text())
        store = cls()
        for u in data["users"]:
            store.upsert_user(User(**u))
        for p in data["posts"]:
            store.upsert_post(Post(**p))
        for k, vs in data.get("connections", {}).items():
            for v in vs:
                store.connect(k, v)
        for e in data["events"]:
            store._events.append(Event(**e))
        return store


_GLOBAL: Store | None = None


def get_store() -> Store:
    """Process-wide singleton, primarily for the API."""
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = Store()
    return _GLOBAL


def set_store(store: Store) -> None:
    global _GLOBAL
    _GLOBAL = store
