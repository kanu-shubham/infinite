import numpy as np
import torch

from ml.feed_ranking.losses import NCETorchLoss, normalized_cross_entropy


def test_nce_is_one_when_predicting_prior():
    rng = np.random.default_rng(0)
    n = 5000
    p = 0.1
    y = (rng.random(n) < p).astype("float32")
    probs = np.full(n, y.mean(), dtype="float32")
    nce = normalized_cross_entropy(probs, y, background_ctr=float(y.mean()))
    assert abs(nce - 1.0) < 1e-3


def test_nce_lower_for_perfect_predictions():
    rng = np.random.default_rng(0)
    y = (rng.random(1000) < 0.1).astype("float32")
    near_perfect = np.where(y > 0.5, 0.99, 0.01).astype("float32")
    prior = np.full_like(near_perfect, y.mean())
    nce_perfect = normalized_cross_entropy(near_perfect, y)
    nce_prior = normalized_cross_entropy(prior, y)
    assert nce_perfect < nce_prior
    assert nce_perfect < 0.2


def test_nce_torch_loss_matches_bce_rescaled():
    bg = 0.1
    loss = NCETorchLoss(bg)
    logits = torch.tensor([0.5, -1.0, 2.0, -0.3])
    y = torch.tensor([1.0, 0.0, 1.0, 0.0])
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    expected = bce / loss.denom
    got = loss(logits, y)
    assert torch.allclose(got, expected)
