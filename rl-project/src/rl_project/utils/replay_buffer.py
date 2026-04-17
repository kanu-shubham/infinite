"""Fixed-size FIFO replay buffer for off-policy algorithms (DQN)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Batch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """A pre-allocated NumPy-backed replay buffer.

    Pre-allocation avoids Python list overhead and gives stable memory usage,
    which matters when you run for millions of steps.
    """

    def __init__(self, capacity: int, obs_shape: tuple[int, ...], seed: int | None = None):
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self._idx = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        i = self._idx
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = float(done)
        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Batch:
        if self._size < batch_size:
            raise ValueError(f"Not enough samples: have {self._size}, need {batch_size}")
        idx = self._rng.integers(0, self._size, size=batch_size)
        return Batch(
            obs=torch.as_tensor(self.obs[idx], device=device),
            actions=torch.as_tensor(self.actions[idx], device=device),
            rewards=torch.as_tensor(self.rewards[idx], device=device),
            next_obs=torch.as_tensor(self.next_obs[idx], device=device),
            dones=torch.as_tensor(self.dones[idx], device=device),
        )
