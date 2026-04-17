from __future__ import annotations

import numpy as np
import pytest
import torch

from rl_project.utils import ReplayBuffer


def test_buffer_fifo_and_size():
    buf = ReplayBuffer(capacity=4, obs_shape=(2,), seed=0)
    for i in range(6):
        buf.add(
            np.array([i, i], dtype=np.float32),
            action=i % 2,
            reward=float(i),
            next_obs=np.array([i + 1, i + 1], dtype=np.float32),
            done=False,
        )
    assert len(buf) == 4
    # After 6 writes into a 4-slot buffer, the oldest two were overwritten.
    assert buf.obs[0, 0] == 4.0
    assert buf.obs[1, 0] == 5.0


def test_sample_shapes():
    buf = ReplayBuffer(capacity=32, obs_shape=(3,), seed=0)
    for i in range(32):
        buf.add(
            np.zeros(3, dtype=np.float32),
            i % 4,
            float(i),
            np.ones(3, dtype=np.float32),
            i % 5 == 0,
        )
    batch = buf.sample(16, device=torch.device("cpu"))
    assert batch.obs.shape == (16, 3)
    assert batch.actions.shape == (16,)
    assert batch.rewards.shape == (16,)
    assert batch.next_obs.shape == (16, 3)
    assert batch.dones.shape == (16,)


def test_sample_raises_when_empty():
    buf = ReplayBuffer(capacity=8, obs_shape=(2,), seed=0)
    with pytest.raises(ValueError):
        buf.sample(4, device=torch.device("cpu"))
