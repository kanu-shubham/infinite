import torch

from ml.feed_ranking.models import build_model


def test_logreg_forward_shape():
    m = build_model("logreg", input_dim=27)
    x = torch.randn(4, 27)
    out = m(x)
    assert out.shape == (4,)


def test_mlp_forward_shape_and_grad():
    m = build_model("mlp", input_dim=27, hidden_dims=(16, 8), dropout=0.0)
    x = torch.randn(4, 27, requires_grad=False)
    out = m(x)
    assert out.shape == (4,)
    loss = out.sum()
    loss.backward()
    assert any(p.grad is not None for p in m.parameters())


def test_unknown_kind_raises():
    import pytest

    with pytest.raises(ValueError):
        build_model("transformer", input_dim=10)
