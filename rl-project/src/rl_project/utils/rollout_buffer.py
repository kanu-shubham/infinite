"""On-policy rollout buffer with GAE advantage computation (PPO)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RolloutBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor


class RolloutBuffer:
    """Stores a fixed-length trajectory, computes GAE advantages and returns.

    Layout: length `T`, `N` parallel environments -> shape (T, N, ...).
    """

    def __init__(
        self,
        rollout_length: int,
        num_envs: int,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        *,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        discrete: bool = True,
    ):
        self.T = rollout_length
        self.N = num_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.discrete = discrete

        self.obs = np.zeros((self.T, self.N, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(
            (self.T, self.N, *action_shape),
            dtype=np.int64 if discrete else np.float32,
        )
        self.log_probs = np.zeros((self.T, self.N), dtype=np.float32)
        self.rewards = np.zeros((self.T, self.N), dtype=np.float32)
        self.values = np.zeros((self.T, self.N), dtype=np.float32)
        self.dones = np.zeros((self.T, self.N), dtype=np.float32)
        self._ptr = 0

    @property
    def full(self) -> bool:
        return self._ptr >= self.T

    def reset(self) -> None:
        self._ptr = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        log_prob: np.ndarray,
        reward: np.ndarray,
        value: np.ndarray,
        done: np.ndarray,
    ) -> None:
        i = self._ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.values[i] = value
        self.dones[i] = done
        self._ptr += 1

    def compute_gae(self, last_values: np.ndarray, last_dones: np.ndarray) -> None:
        """Compute GAE advantages and bootstrapped returns.

        Parameters
        ----------
        last_values : shape (N,)
            V(s_T) from the critic, used to bootstrap the tail.
        last_dones : shape (N,)
            Whether the env was done right after the last step.
        """
        advantages = np.zeros_like(self.rewards)
        gae = np.zeros(self.N, dtype=np.float32)
        for t in reversed(range(self.T)):
            if t == self.T - 1:
                next_non_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]
            delta = self.rewards[t] + self.gamma * next_values * next_non_terminal - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
        self.advantages = advantages
        self.returns = advantages + self.values

    def iter_minibatches(self, minibatch_size: int, device: torch.device):
        """Yield RolloutBatch minibatches over the flattened buffer."""
        batch_size = self.T * self.N
        obs = self.obs.reshape(batch_size, *self.obs.shape[2:])
        actions = self.actions.reshape(batch_size, *self.actions.shape[2:])
        log_probs = self.log_probs.reshape(batch_size)
        advantages = self.advantages.reshape(batch_size)
        returns = self.returns.reshape(batch_size)
        values = self.values.reshape(batch_size)

        idx = np.random.permutation(batch_size)
        for start in range(0, batch_size, minibatch_size):
            mb = idx[start:start + minibatch_size]
            yield RolloutBatch(
                obs=torch.as_tensor(obs[mb], device=device),
                actions=torch.as_tensor(actions[mb], device=device),
                old_log_probs=torch.as_tensor(log_probs[mb], device=device),
                advantages=torch.as_tensor(advantages[mb], device=device),
                returns=torch.as_tensor(returns[mb], device=device),
                values=torch.as_tensor(values[mb], device=device),
            )
