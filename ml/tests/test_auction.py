import numpy as np

from ml.models.auction import run_auction


def test_top_slot_pays_second_score_over_own_effective():
    pctr = np.array([0.10, 0.08, 0.04], dtype=np.float32)
    bids = np.array([1.0, 1.5, 2.0], dtype=np.float32)
    quality = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    # rank scores: 0.10, 0.12, 0.08 -> winner is idx 1 (0.12), next 0.10
    res = run_auction(pctr, bids, quality, n_slots=2, reserve_cpc=0.01)
    assert list(res.order) == [1, 0]
    expected_price_top = (0.10 / (0.08 * 1.0)) + 0.01  # second_score / (pctr*quality) + eps
    assert abs(res.prices[0] - expected_price_top) < 1e-4


def test_reserve_floor_applied():
    pctr = np.array([0.5, 0.4], dtype=np.float32)
    bids = np.array([0.10, 0.09], dtype=np.float32)
    quality = np.array([1.0, 1.0], dtype=np.float32)
    res = run_auction(pctr, bids, quality, n_slots=2, reserve_cpc=0.20)
    assert all(p >= 0.20 for p in res.prices)


def test_price_never_exceeds_bid():
    rng = np.random.default_rng(0)
    n = 50
    pctr = rng.uniform(0.01, 0.2, n).astype(np.float32)
    bids = rng.uniform(0.1, 3.0, n).astype(np.float32)
    q = rng.uniform(0.3, 1.0, n).astype(np.float32)
    res = run_auction(pctr, bids, q, n_slots=10, reserve_cpc=0.05)
    assert (res.prices <= bids[res.order] + 1e-6).all()


def test_n_slots_capped_to_candidate_count():
    pctr = np.array([0.1], dtype=np.float32)
    bids = np.array([1.0], dtype=np.float32)
    q = np.array([1.0], dtype=np.float32)
    res = run_auction(pctr, bids, q, n_slots=8)
    assert len(res.order) == 1


def test_empty_candidates():
    z = np.empty(0, dtype=np.float32)
    res = run_auction(z, z, z, n_slots=8)
    assert len(res.order) == 0
