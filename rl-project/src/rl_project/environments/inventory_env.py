"""Smart Warehouse Inventory Management environment.

An RL-ready Gymnasium environment that simulates one SKU in a warehouse. Each
step is one business day. The agent chooses how many units to order from the
supplier; the order arrives next morning. Demand is stochastic and depends on
the day of the week plus a random non-stationary drift to make the problem
realistic.

MDP summary (see docs/05-real-world-inventory.md for the full story):

- State (5 dims, normalized):
    stock_on_hand, on_order, day_of_week, recent_demand_mean, recent_demand_std
- Action:
    discrete: index into ORDER_CHOICES
    continuous: scalar in [0, 1] mapped to [0, capacity]
- Reward (dollars):
    price * sales - cost * order - holding * new_stock - stockout_penalty * lost
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Discrete order choices used when action_mode="discrete" (DQN).
ORDER_CHOICES: tuple[int, ...] = (0, 5, 10, 20, 30, 50)


@dataclass
class InventoryConfig:
    """Economic and simulation parameters for the warehouse."""

    capacity: int = 100
    horizon: int = 60
    # Economics (dollars).
    price: float = 10.0
    order_cost: float = 4.0
    holding_cost: float = 0.2
    stockout_penalty: float = 6.0
    # Demand model: per-weekday Poisson means, plus a slow random drift.
    base_demand_by_weekday: tuple[float, ...] = (
        12.0, 14.0, 15.0, 16.0, 20.0, 28.0, 24.0,
    )
    demand_drift_std: float = 0.05  # random-walk on the multiplier.
    # Rolling window for observation features.
    rolling_window: int = 7
    # Starting stock sampled from [0, init_stock_max].
    init_stock_max: int = 40


@dataclass
class _State:
    stock: int = 0
    on_order: int = 0
    day: int = 0
    demand_multiplier: float = 1.0
    recent_demand: deque = field(default_factory=lambda: deque(maxlen=7))


class InventoryEnv(gym.Env):
    """A Gymnasium environment for single-SKU inventory replenishment.

    Parameters
    ----------
    config : InventoryConfig, optional
        Economic and simulation parameters.
    action_mode : {"discrete", "continuous"}
        - "discrete" -> Discrete(len(ORDER_CHOICES)) for DQN.
        - "continuous" -> Box([0], [1]) mapped to [0, capacity] for PPO.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: InventoryConfig | None = None,
        action_mode: str = "continuous",
    ):
        super().__init__()
        self.config = config or InventoryConfig()
        if action_mode not in {"discrete", "continuous"}:
            raise ValueError(f"Unknown action_mode={action_mode!r}")
        self.action_mode = action_mode

        if action_mode == "discrete":
            self.action_space = spaces.Discrete(len(ORDER_CHOICES))
        else:
            self.action_space = spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32,
            )

        # Observation: normalized to roughly [-1, 1] range for NN stability.
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(5,), dtype=np.float32,
        )

        self._state = _State(recent_demand=deque(maxlen=self.config.rolling_window))
        self._t = 0

    # ---------- Gym API ----------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        rng = self.np_random
        self._state = _State(
            stock=int(rng.integers(0, self.config.init_stock_max + 1)),
            on_order=0,
            day=int(rng.integers(0, 7)),
            demand_multiplier=1.0,
            recent_demand=deque(
                [float(np.mean(self.config.base_demand_by_weekday))] * self.config.rolling_window,
                maxlen=self.config.rolling_window,
            ),
        )
        self._t = 0
        return self._obs(), {}

    def step(
        self, action: int | np.ndarray | float,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        order = self._action_to_order(action)

        # Morning: yesterday's on_order arrives.
        stock_after_arrival = min(self._state.stock + self._state.on_order, self.config.capacity)

        # Daytime: sample demand for today.
        demand = self._sample_demand()
        sales = min(stock_after_arrival, demand)
        lost = max(demand - stock_after_arrival, 0)
        new_stock = stock_after_arrival - sales

        # Reward (dollars).
        reward = (
            self.config.price * sales
            - self.config.order_cost * order
            - self.config.holding_cost * new_stock
            - self.config.stockout_penalty * lost
        )

        # Transition.
        self._state.stock = new_stock
        self._state.on_order = order
        self._state.day = (self._state.day + 1) % 7
        self._state.recent_demand.append(float(demand))
        # Slow random-walk on demand multiplier to make the task non-stationary.
        drift = float(self.np_random.normal(0.0, self.config.demand_drift_std))
        self._state.demand_multiplier = float(
            np.clip(self._state.demand_multiplier * np.exp(drift), 0.5, 2.0),
        )
        self._t += 1

        terminated = False
        truncated = self._t >= self.config.horizon

        info = {
            "demand": int(demand),
            "sales": int(sales),
            "lost_sales": int(lost),
            "order": int(order),
            "stock": int(new_stock),
            "reward_breakdown": {
                "revenue": self.config.price * sales,
                "order_cost": self.config.order_cost * order,
                "holding_cost": self.config.holding_cost * new_stock,
                "stockout_penalty": self.config.stockout_penalty * lost,
            },
        }
        return self._obs(), float(reward), terminated, truncated, info

    def render(self) -> None:  # pragma: no cover - trivial
        print(
            f"t={self._t:3d} day={self._state.day} "
            f"stock={self._state.stock:3d} on_order={self._state.on_order:3d} "
            f"mult={self._state.demand_multiplier:.2f}",
        )

    # ---------- helpers ----------

    def _action_to_order(self, action: int | np.ndarray | float) -> int:
        if self.action_mode == "discrete":
            idx = int(action)
            if idx < 0 or idx >= len(ORDER_CHOICES):
                raise ValueError(f"Discrete action out of range: {idx}")
            order = ORDER_CHOICES[idx]
        else:
            a = float(np.asarray(action).reshape(-1)[0])
            a = float(np.clip(a, 0.0, 1.0))
            order = int(round(a * self.config.capacity))
        # Respect capacity: can't have stock+on_order+order beyond capacity.
        max_order = max(0, self.config.capacity - self._state.stock - self._state.on_order)
        return min(order, max_order)

    def _sample_demand(self) -> int:
        base = self.config.base_demand_by_weekday[self._state.day]
        mean = base * self._state.demand_multiplier
        return int(self.np_random.poisson(mean))

    def _obs(self) -> np.ndarray:
        cap = float(self.config.capacity)
        mean_base = float(np.mean(self.config.base_demand_by_weekday))
        recent = np.asarray(self._state.recent_demand, dtype=np.float32)
        rec_mean = float(recent.mean()) if recent.size else mean_base
        rec_std = float(recent.std()) if recent.size else 0.0
        obs = np.array(
            [
                self._state.stock / cap,                # in [0, 1]
                self._state.on_order / cap,             # in [0, 1]
                self._state.day / 6.0,                  # in [0, 1]
                (rec_mean - mean_base) / mean_base,     # ~[-1, 1]
                rec_std / mean_base,                    # ~[0, 1]
            ],
            dtype=np.float32,
        )
        return obs
