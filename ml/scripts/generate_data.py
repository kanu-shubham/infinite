"""Seed an in-memory store with synthetic users/posts/events and persist it."""
from __future__ import annotations

import argparse

from ml.feed_ranking.config import DEFAULT_DATASET
from ml.feed_ranking.data import GenerationConfig, generate_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--users", type=int, default=500)
    p.add_argument("--posts", type=int, default=2000)
    p.add_argument("--impressions", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--out", type=str, default=str(DEFAULT_DATASET))
    args = p.parse_args()

    cfg = GenerationConfig(
        n_users=args.users,
        n_posts=args.posts,
        n_impressions=args.impressions,
        seed=args.seed,
    )
    store = generate_dataset(cfg)
    out_path = type(DEFAULT_DATASET)(args.out)
    store.save_json(out_path)
    n_clicks = sum(1 for e in store.all_events() if e.clicked)
    print(
        f"wrote {out_path}: users={len(store.all_users())} "
        f"posts={len(store.all_posts())} events={len(store.all_events())} "
        f"clicks={n_clicks} ctr={n_clicks/len(store.all_events()):.4f}"
    )


if __name__ == "__main__":
    main()
