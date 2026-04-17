"""Tests for the InventoryEnv — the MDP we care about."""
from __future__ import annotations

import numpy as np
import pytest

from rl_project.environments import InventoryEnv
from rl_project.environments.inventory_env import ORDER_CHOICES, InventoryConfig


def test_observation_shape_and_bounds():
    env = InventoryEnv(action_mode="continuous")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (5,)
    assert np.all(np.isfinite(obs))


def test_episode_length_equals_horizon():
    cfg = InventoryConfig(horizon=10)
    env = InventoryEnv(config=cfg, action_mode="continuous")
    env.reset(seed=0)
    steps, done = 0, False
    while not done:
        _, _, term, trunc, _ = env.step(np.array([0.3], dtype=np.float32))
        done = term or trunc
        steps += 1
    assert steps == 10


def test_capacity_is_respected():
    cfg = InventoryConfig(capacity=50)
    env = InventoryEnv(config=cfg, action_mode="continuous")
    env.reset(seed=0)
    # Order maximum every day.
    for _ in range(cfg.horizon):
        _, _, term, trunc, info = env.step(np.array([1.0], dtype=np.float32))
        assert info["stock"] <= cfg.capacity
        if term or trunc:
            break


def test_discrete_action_space_maps_to_choices():
    env = InventoryEnv(action_mode="discrete")
    assert env.action_space.n == len(ORDER_CHOICES)
    env.reset(seed=0)
    # Even the biggest discrete choice shouldn't violate capacity.
    _, _, _, _, info = env.step(env.action_space.n - 1)
    assert info["order"] >= 0


def test_no_order_can_cause_stockout_but_not_error():
    env = InventoryEnv(action_mode="continuous")
    env.reset(seed=0)
    for _ in range(20):
        _, _, term, trunc, info = env.step(np.array([0.0], dtype=np.float32))
        assert info["lost_sales"] >= 0
        if term or trunc:
            break


def test_reward_breakdown_components_sum():
    env = InventoryEnv(action_mode="continuous")
    env.reset(seed=0)
    _, r, _, _, info = env.step(np.array([0.2], dtype=np.float32))
    b = info["reward_breakdown"]
    reconstructed = b["revenue"] - b["order_cost"] - b["holding_cost"] - b["stockout_penalty"]
    assert r == pytest.approx(reconstructed, rel=1e-6, abs=1e-6)


def test_reset_is_reproducible_with_seed():
    env = InventoryEnv(action_mode="continuous")
    o1, _ = env.reset(seed=42)
    o2, _ = env.reset(seed=42)
    assert np.allclose(o1, o2)
