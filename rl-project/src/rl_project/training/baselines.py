"""Classical (non-RL) inventory policies to compare against.

These are what an operations-research team would build without RL. Any RL
agent worth deploying must beat them on the same demand sequences.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl_project.environments import InventoryEnv
from rl_project.environments.inventory_env import ORDER_CHOICES


class _Policy:
    """Duck-typed policy interface used by `evaluate_policy`."""

    def act(self, env: InventoryEnv, obs: np.ndarray) -> int | np.ndarray:
        raise NotImplementedError


class DoNothingPolicy(_Policy):
    """Never order anything. Sanity-check floor."""

    def act(self, env: InventoryEnv, obs: np.ndarray):
        if env.action_mode == "discrete":
            return 0
        return np.array([0.0], dtype=np.float32)


@dataclass
class ReorderPointPolicy(_Policy):
    """Classic (s, S) policy: if stock+on_order <= reorder_point, order up to target."""

    reorder_point: int = 20
    target_level: int = 60

    def act(self, env: InventoryEnv, obs: np.ndarray):
        stock = env._state.stock + env._state.on_order  # type: ignore[attr-defined]
        if stock <= self.reorder_point:
            qty = max(0, self.target_level - stock)
        else:
            qty = 0
        return self._quantize(env, qty)

    def _quantize(self, env: InventoryEnv, qty: int):
        if env.action_mode == "discrete":
            # Pick the closest discrete order choice.
            best = min(range(len(ORDER_CHOICES)), key=lambda i: abs(ORDER_CHOICES[i] - qty))
            return best
        # Continuous: scalar in [0, 1].
        return np.array([qty / env.config.capacity], dtype=np.float32)


@dataclass
class NewsvendorPolicy(_Policy):
    """Newsvendor-inspired policy: target a service level and bring stock to it.

    Order-up-to level Q* = mu + z * sigma, where z corresponds to the critical
    ratio p / (p + h), computed from the env's economics.
    """

    def __post_init__(self) -> None:
        self._z_cache: float | None = None

    def _z(self, env: InventoryEnv) -> float:
        # Critical ratio for newsvendor.
        p = env.config.stockout_penalty
        h = env.config.holding_cost
        # Normal inverse CDF for F^-1(p / (p + h)); cheap approximation.
        ratio = p / (p + h)
        # Rational approximation to qnorm (Beasley-Springer-Moro is overkill here).
        return float(np.sqrt(2) * _erfinv(2 * ratio - 1))

    def act(self, env: InventoryEnv, obs: np.ndarray):
        recent = np.asarray(env._state.recent_demand, dtype=np.float32)  # type: ignore[attr-defined]
        mu = float(recent.mean()) if recent.size else float(np.mean(env.config.base_demand_by_weekday))
        sigma = float(recent.std()) if recent.size else 5.0
        target = mu + self._z(env) * sigma
        pipeline = env._state.stock + env._state.on_order  # type: ignore[attr-defined]
        qty = max(0, int(round(target - pipeline)))
        if env.action_mode == "discrete":
            best = min(range(len(ORDER_CHOICES)), key=lambda i: abs(ORDER_CHOICES[i] - qty))
            return best
        return np.array([qty / env.config.capacity], dtype=np.float32)


def _erfinv(x: float) -> float:
    """Inverse error function via Winitzki approximation (plenty accurate here)."""
    a = 0.147
    ln = np.log(1 - x * x + 1e-12)
    first = 2 / (np.pi * a) + ln / 2
    return float(np.sign(x) * np.sqrt(np.sqrt(first * first - ln / a) - first))


def evaluate_policy(
    policy: _Policy,
    env: InventoryEnv,
    episodes: int = 20,
    seed: int = 0,
) -> dict[str, float]:
    """Roll out the policy for `episodes` episodes and return aggregate metrics."""
    returns: list[float] = []
    stockouts: list[int] = []
    holding: list[float] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_ret = 0.0
        ep_stockout = 0
        ep_hold = 0.0
        done = False
        while not done:
            action = policy.act(env, obs)
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            ep_ret += r
            ep_stockout += int(info["lost_sales"])
            ep_hold += float(info["reward_breakdown"]["holding_cost"])
        returns.append(ep_ret)
        stockouts.append(ep_stockout)
        holding.append(ep_hold)
    return {
        "profit_mean": float(np.mean(returns)),
        "profit_std": float(np.std(returns)),
        "stockout_units_mean": float(np.mean(stockouts)),
        "holding_cost_mean": float(np.mean(holding)),
    }
