"""Neural network building blocks used by DQN and PPO."""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical, Normal


def _orthogonal_init(layer: nn.Linear, gain: float = 1.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class MLP(nn.Module):
    """Simple multi-layer perceptron with tanh activations (PPO-friendly)."""

    def __init__(self, in_dim: int, out_dim: int, hidden: tuple[int, ...] = (64, 64), gain: float = 1.0):
        super().__init__()
        layers: list[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers.append(_orthogonal_init(nn.Linear(last, h), gain=2 ** 0.5))
            layers.append(nn.Tanh())
            last = h
        layers.append(_orthogonal_init(nn.Linear(last, out_dim), gain=gain))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QNetwork(nn.Module):
    """Q(s, .) network for DQN. Outputs one value per discrete action."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: tuple[int, ...] = (128, 128)):
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCritic(nn.Module):
    """Actor-critic network for PPO.

    Supports both discrete (Categorical) and continuous (diagonal Normal)
    action spaces. The two heads share the input but not hidden layers, which
    is the standard choice in PPO implementations (better gradient flow).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        discrete: bool,
        hidden: tuple[int, ...] = (64, 64),
    ):
        super().__init__()
        self.discrete = discrete
        self.action_dim = action_dim
        self.actor = MLP(obs_dim, action_dim, hidden=hidden, gain=0.01)
        self.critic = MLP(obs_dim, 1, hidden=hidden, gain=1.0)
        if not discrete:
            # State-independent log std, learned. Standard PPO trick.
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor):
        logits_or_mean = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        if self.discrete:
            dist = Categorical(logits=logits_or_mean)
        else:
            std = self.log_std.exp().expand_as(logits_or_mean)
            dist = Normal(logits_or_mean, std)
        return dist, value

    def act(self, obs: torch.Tensor):
        """Sample an action with log prob and state value."""
        dist, value = self.forward(obs)
        action = dist.sample()
        if self.discrete:
            log_prob = dist.log_prob(action)
        else:
            # Sum log probs over independent dims for diagonal Gaussian.
            log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        """Evaluate log prob, entropy, and value of given actions."""
        dist, value = self.forward(obs)
        if self.discrete:
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        else:
            log_prob = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value
