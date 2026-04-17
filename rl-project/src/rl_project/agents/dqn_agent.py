"""Deep Q-Network (DQN) agent.

Implements vanilla DQN with two well-established stabilizers:

* **Experience replay buffer** — decorrelates samples, improves efficiency.
* **Target network with soft (Polyak) updates** — stabilizes the TD target.
* **Double DQN** — online net selects the action, target net evaluates it.

See docs/03-dqn-explained.md for the theory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from rl_project.networks import QNetwork
from rl_project.utils import ReplayBuffer


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 128
    buffer_size: int = 100_000
    min_buffer: int = 1_000
    target_tau: float = 0.005            # soft-update coefficient
    train_every: int = 1                 # gradient step every N env steps
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000
    grad_clip: float = 10.0
    hidden: tuple[int, ...] = field(default_factory=lambda: (128, 128))
    double_dqn: bool = True


class DQNAgent:
    """DQN agent for discrete action spaces."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        config: DQNConfig | None = None,
        *,
        device: str | torch.device | None = None,
        seed: int | None = None,
    ):
        self.config = config or DQNConfig()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"),
        )
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.q = QNetwork(obs_dim, n_actions, hidden=self.config.hidden).to(self.device)
        self.q_target = QNetwork(obs_dim, n_actions, hidden=self.config.hidden).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        for p in self.q_target.parameters():
            p.requires_grad_(False)

        self.optimizer = optim.Adam(self.q.parameters(), lr=self.config.lr)
        self.loss_fn = nn.SmoothL1Loss()        # Huber — robust to outliers.
        self.buffer = ReplayBuffer(
            capacity=self.config.buffer_size,
            obs_shape=(obs_dim,),
            seed=seed,
        )
        self._steps = 0

    # ---------- action selection ----------

    def epsilon(self) -> float:
        c = self.config
        frac = min(1.0, self._steps / max(1, c.epsilon_decay_steps))
        return c.epsilon_start + frac * (c.epsilon_end - c.epsilon_start)

    def select_action(self, obs: np.ndarray, *, greedy: bool = False) -> int:
        if not greedy and np.random.random() < self.epsilon():
            return int(np.random.randint(self.n_actions))
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q(obs_t)
            return int(q.argmax(dim=1).item())

    # ---------- learning ----------

    def observe(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.add(obs, action, reward, next_obs, done)
        self._steps += 1

    def update(self) -> dict[str, float] | None:
        c = self.config
        if len(self.buffer) < max(c.batch_size, c.min_buffer):
            return None
        if self._steps % c.train_every != 0:
            return None

        batch = self.buffer.sample(c.batch_size, device=self.device)
        # Current Q(s, a)
        q_values = self.q(batch.obs).gather(1, batch.actions.unsqueeze(1)).squeeze(1)

        # Target y = r + gamma * (1 - done) * Q_target(s', argmax_a Q(s', a))
        with torch.no_grad():
            if c.double_dqn:
                next_actions = self.q(batch.next_obs).argmax(dim=1, keepdim=True)
                next_q = self.q_target(batch.next_obs).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.q_target(batch.next_obs).max(dim=1).values
            target = batch.rewards + c.gamma * (1.0 - batch.dones) * next_q

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), c.grad_clip)
        self.optimizer.step()

        # Soft target update: theta_target <- tau*theta + (1-tau)*theta_target
        with torch.no_grad():
            for p, p_t in zip(self.q.parameters(), self.q_target.parameters()):
                p_t.data.mul_(1.0 - c.target_tau).add_(c.target_tau * p.data)

        return {
            "loss": float(loss.item()),
            "q_mean": float(q_values.mean().item()),
            "epsilon": self.epsilon(),
        }

    # ---------- persistence ----------

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q": self.q.state_dict(),
                "q_target": self.q_target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.config.__dict__,
                "steps": self._steps,
            },
            p,
        )

    def load(self, path: str | Path) -> None:
        blob = torch.load(path, map_location=self.device)
        self.q.load_state_dict(blob["q"])
        self.q_target.load_state_dict(blob["q_target"])
        self.optimizer.load_state_dict(blob["optimizer"])
        self._steps = int(blob.get("steps", 0))
