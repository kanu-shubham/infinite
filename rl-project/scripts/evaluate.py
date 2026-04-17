"""Evaluate a trained agent and compare against rule-based baselines."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rl_project.agents import DQNAgent, PPOAgent
from rl_project.environments import InventoryEnv
from rl_project.environments.inventory_env import InventoryConfig
from rl_project.training.baselines import (
    DoNothingPolicy,
    NewsvendorPolicy,
    ReorderPointPolicy,
    evaluate_policy,
)


class _DQNAdapter:
    def __init__(self, agent: DQNAgent):
        self.agent = agent

    def act(self, env, obs):
        return self.agent.select_action(obs, greedy=True)


class _PPOAdapter:
    def __init__(self, agent: PPOAgent):
        self.agent = agent

    def act(self, env, obs):
        return self.agent.greedy_action(obs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["dqn", "ppo"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    action_mode = "discrete" if args.algo == "dqn" else "continuous"
    env = InventoryEnv(config=InventoryConfig(), action_mode=action_mode)

    obs_dim = int(np.prod(env.observation_space.shape))
    if args.algo == "dqn":
        agent = DQNAgent(obs_dim, int(env.action_space.n))
        agent.load(args.checkpoint)
        policy = _DQNAdapter(agent)
    else:
        action_dim = int(np.prod(env.action_space.shape))
        agent = PPOAgent(obs_dim, action_dim, discrete=False)
        agent.load(args.checkpoint)
        policy = _PPOAdapter(agent)

    print("Evaluating on held-out demand sequences...")
    results = {
        "RL agent": evaluate_policy(policy, env, episodes=args.episodes, seed=args.seed),
        "Do nothing": evaluate_policy(DoNothingPolicy(), env, episodes=args.episodes, seed=args.seed),
        "(s=20, S=60) reorder": evaluate_policy(
            ReorderPointPolicy(reorder_point=20, target_level=60),
            env, episodes=args.episodes, seed=args.seed,
        ),
        "Newsvendor": evaluate_policy(NewsvendorPolicy(), env, episodes=args.episodes, seed=args.seed),
    }

    print(f"\n{'Policy':<25} {'Profit':>12} {'±':>10} {'Stockouts':>12} {'Holding':>10}")
    print("-" * 75)
    for name, r in results.items():
        print(
            f"{name:<25} {r['profit_mean']:>12.1f} {r['profit_std']:>10.1f} "
            f"{r['stockout_units_mean']:>12.1f} {r['holding_cost_mean']:>10.2f}",
        )


if __name__ == "__main__":
    main()
