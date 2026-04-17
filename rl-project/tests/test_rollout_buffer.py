from __future__ import annotations

import numpy as np
import torch

from rl_project.utils import RolloutBuffer


def test_gae_matches_closed_form_with_zero_values():
    """With V(s)=0 and no terminations, GAE reduces to discounted sum of rewards."""
    T, N = 5, 2
    buf = RolloutBuffer(
        rollout_length=T,
        num_envs=N,
        obs_shape=(3,),
        action_shape=(),
        gamma=0.9,
        gae_lambda=1.0,  # Monte Carlo.
        discrete=True,
    )
    for t in range(T):
        buf.add(
            obs=np.zeros((N, 3), dtype=np.float32),
            action=np.zeros(N, dtype=np.int64),
            log_prob=np.zeros(N, dtype=np.float32),
            reward=np.ones(N, dtype=np.float32),  # reward=1 each step
            value=np.zeros(N, dtype=np.float32),
            done=np.zeros(N, dtype=np.float32),
        )
    buf.compute_gae(last_values=np.zeros(N, dtype=np.float32), last_dones=np.zeros(N, dtype=np.float32))
    # Advantage at step t with lambda=1, V=0: sum_{k=0..T-1-t} gamma^k * 1
    expected_adv_t0 = sum(0.9 ** k for k in range(T))
    assert buf.advantages[0, 0] == np.float32(expected_adv_t0)


def test_iter_minibatches_covers_all_samples():
    T, N = 4, 3
    buf = RolloutBuffer(T, N, (2,), (), gamma=0.99, gae_lambda=0.95, discrete=True)
    for t in range(T):
        buf.add(
            np.ones((N, 2), dtype=np.float32) * t,
            np.zeros(N, dtype=np.int64),
            np.zeros(N, dtype=np.float32),
            np.zeros(N, dtype=np.float32),
            np.zeros(N, dtype=np.float32),
            np.zeros(N, dtype=np.float32),
        )
    buf.compute_gae(np.zeros(N, dtype=np.float32), np.zeros(N, dtype=np.float32))
    seen = 0
    for mb in buf.iter_minibatches(4, device=torch.device("cpu")):
        seen += mb.obs.shape[0]
    assert seen == T * N
