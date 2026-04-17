"""Proximal Policy Optimization (PPO) agent.

Implements the clipped-objective actor-critic variant from Schulman et al. 2017
with Generalized Advantage Estimation.

See docs/04-ppo-explained.md for the theory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from rl_project.networks import ActorCritic
from rl_project.utils import RolloutBuffer


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    lr: float = 3e-4
    num_envs: int = 8
    rollout_length: int = 128
    epochs: int = 10
    minibatch_size: int = 64
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    grad_clip: float = 0.5
    target_kl: float | None = 0.02       # early-stop per update if exceeded
    hidden: tuple[int, ...] = field(default_factory=lambda: (64, 64))
    normalize_advantages: bool = True


class PPOAgent:
    """PPO agent for discrete or continuous actions."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        discrete: bool,
        config: PPOConfig | None = None,
        device: str | torch.device | None = None,
        seed: int | None = None,
    ):
        self.config = config or PPOConfig()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"),
        )
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.net = ActorCritic(
            obs_dim, action_dim, discrete=discrete, hidden=self.config.hidden,
        ).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.config.lr, eps=1e-5)

        action_shape: tuple[int, ...] = () if discrete else (action_dim,)
        self.buffer = RolloutBuffer(
            rollout_length=self.config.rollout_length,
            num_envs=self.config.num_envs,
            obs_shape=(obs_dim,),
            action_shape=action_shape,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            discrete=discrete,
        )

    # ---------- action selection ----------

    @torch.no_grad()
    def act(self, obs: np.ndarray):
        """Sample actions for a batch of environments.

        Parameters
        ----------
        obs : ndarray shape (N, obs_dim)

        Returns
        -------
        actions : ndarray
        log_probs : ndarray shape (N,)
        values : ndarray shape (N,)
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        action, log_prob, value = self.net.act(obs_t)
        return (
            action.cpu().numpy(),
            log_prob.cpu().numpy(),
            value.cpu().numpy(),
        )

    @torch.no_grad()
    def value(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        _, v = self.net(obs_t)
        return v.cpu().numpy()

    @torch.no_grad()
    def greedy_action(self, obs: np.ndarray):
        """Deterministic action: argmax logits (discrete) or mean (continuous)."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        dist, _ = self.net(obs_t)
        if self.discrete:
            a = dist.probs.argmax(dim=-1)
        else:
            a = dist.mean
        a = a.cpu().numpy()
        return a[0] if squeeze else a

    # ---------- learning ----------

    def update(self) -> dict[str, float]:
        c = self.config
        # Flatten and normalize advantages.
        if c.normalize_advantages:
            adv = self.buffer.advantages
            self.buffer.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

        stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_frac": 0.0,
            "updates": 0,
        }

        early_stop = False
        for _epoch in range(c.epochs):
            if early_stop:
                break
            for batch in self.buffer.iter_minibatches(c.minibatch_size, self.device):
                log_prob, entropy, value = self.net.evaluate(batch.obs, batch.actions)
                ratio = torch.exp(log_prob - batch.old_log_probs)
                surr1 = ratio * batch.advantages
                surr2 = torch.clamp(ratio, 1 - c.clip_eps, 1 + c.clip_eps) * batch.advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (value - batch.returns).pow(2).mean()
                entropy_loss = -entropy.mean()
                loss = (
                    policy_loss
                    + c.value_coef * value_loss
                    + c.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), c.grad_clip)
                self.optimizer.step()

                # Diagnostics.
                with torch.no_grad():
                    approx_kl = (batch.old_log_probs - log_prob).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > c.clip_eps).float().mean().item()

                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy.mean().item())
                stats["approx_kl"] += approx_kl
                stats["clip_frac"] += clip_frac
                stats["updates"] += 1

                if c.target_kl is not None and approx_kl > 1.5 * c.target_kl:
                    early_stop = True
                    break

        self.buffer.reset()
        n = max(1, stats["updates"])
        for k in ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"]:
            stats[k] /= n
        return stats

    # ---------- persistence ----------

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "net": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.config.__dict__,
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "discrete": self.discrete,
            },
            p,
        )

    def load(self, path: str | Path) -> None:
        blob = torch.load(path, map_location=self.device)
        self.net.load_state_dict(blob["net"])
        self.optimizer.load_state_dict(blob["optimizer"])
