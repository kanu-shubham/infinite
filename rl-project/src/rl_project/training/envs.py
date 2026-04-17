"""Environment factories for training and evaluation."""
from __future__ import annotations

from typing import Callable

import gymnasium as gym

from rl_project.environments import InventoryEnv
from rl_project.environments.inventory_env import InventoryConfig


def make_env(
    name: str,
    *,
    action_mode: str = "continuous",
    horizon: int | None = None,
    seed: int | None = None,
) -> gym.Env:
    """Build a single env by name. Supports 'inventory', 'cartpole', 'lunarlander'."""
    name = name.lower()
    if name in {"inventory", "inv"}:
        cfg = InventoryConfig()
        if horizon is not None:
            cfg.horizon = horizon
        env = InventoryEnv(config=cfg, action_mode=action_mode)
    elif name in {"cartpole", "cartpole-v1"}:
        env = gym.make("CartPole-v1")
    elif name in {"lunarlander", "lunarlander-v3"}:
        env = gym.make("LunarLander-v3")
    else:
        env = gym.make(name)
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_vec_env(
    name: str,
    num_envs: int,
    *,
    action_mode: str = "continuous",
    horizon: int | None = None,
    seed: int = 0,
) -> gym.vector.SyncVectorEnv:
    """Build N parallel envs (synchronous) for on-policy rollouts."""

    def factory(rank: int) -> Callable[[], gym.Env]:
        def _thunk() -> gym.Env:
            env = make_env(name, action_mode=action_mode, horizon=horizon, seed=seed + rank)
            return env
        return _thunk

    return gym.vector.SyncVectorEnv([factory(i) for i in range(num_envs)])
