from ml.feed_ranking.data import (
    GenerationConfig,
    downsample_negatives,
    generate_dataset,
    temporal_split,
)


def test_generated_dataset_has_both_classes():
    cfg = GenerationConfig(n_users=20, n_posts=80, n_impressions=2000, seed=1)
    store = generate_dataset(cfg)
    events = store.all_events()
    n_clicks = sum(1 for e in events if e.clicked)
    assert 0 < n_clicks < len(events), f"degenerate label distribution: {n_clicks}/{len(events)}"


def test_temporal_split_orders_by_time():
    cfg = GenerationConfig(n_users=10, n_posts=20, n_impressions=200, seed=2)
    store = generate_dataset(cfg)
    train, val, test = temporal_split(store.all_events(), val_fraction=0.2, test_fraction=0.2)
    assert len(train) + len(val) + len(test) == len(store.all_events())
    if train and val:
        assert max(e.timestamp for e in train) <= min(e.timestamp for e in val)
    if val and test:
        assert max(e.timestamp for e in val) <= min(e.timestamp for e in test)


def test_downsample_keeps_all_positives_and_caps_negatives():
    cfg = GenerationConfig(n_users=10, n_posts=20, n_impressions=500, seed=3)
    store = generate_dataset(cfg)
    events = store.all_events()
    pos = [e for e in events if e.clicked]
    out = downsample_negatives(events, ratio=2.0)
    assert sum(1 for e in out if e.clicked) == len(pos)
    neg_kept = sum(1 for e in out if not e.clicked)
    assert neg_kept <= 2 * len(pos)
