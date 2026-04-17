"""Smoke tests: can the agents construct, take actions, and do one update?"""
from __future__ import annotations

import numpy as np

from rl_project.agents import DQNAgent, DQNConfig, PPOAgent, PPOConfig


def test_dqn_can_take_one_update():
    agent = DQNAgent(
        obs_dim=4,
        n_actions=2,
        config=DQNConfig(min_buffer=4, batch_size=4, epsilon_decay_steps=10),
        seed=0,
    )
    for _ in range(16):
        obs = np.random.randn(4).astype(np.float32)
        next_obs = np.random.randn(4).astype(np.float32)
        a = agent.select_action(obs)
        agent.observe(obs, a, 1.0, next_obs, False)
    stats = agent.update()
    assert stats is not None
    assert np.isfinite(stats["loss"])


def test_ppo_can_collect_and_update():
    cfg = PPOConfig(num_envs=2, rollout_length=4, epochs=1, minibatch_size=4)
    agent = PPOAgent(obs_dim=3, action_dim=2, discrete=True, config=cfg, seed=0)
    obs = np.random.randn(cfg.num_envs, 3).astype(np.float32)
    dones = np.zeros(cfg.num_envs, dtype=np.float32)
    for _ in range(cfg.rollout_length):
        a, lp, v = agent.act(obs)
        r = np.random.randn(cfg.num_envs).astype(np.float32)
        agent.buffer.add(obs, a, lp, r, v, dones)
        obs = np.random.randn(cfg.num_envs, 3).astype(np.float32)
    agent.buffer.compute_gae(
        last_values=np.zeros(cfg.num_envs, dtype=np.float32),
        last_dones=np.zeros(cfg.num_envs, dtype=np.float32),
    )
    stats = agent.update()
    assert np.isfinite(stats["policy_loss"])
    assert np.isfinite(stats["value_loss"])
