"""Tutorial 3: Train PPO on the Smart Warehouse Inventory environment.

This is the end-to-end production story. After training, we evaluate the
learned policy on unseen demand sequences and compare it with three classical
baselines so you can see whether RL actually helps.

Run:
    python tutorials/03_inventory_ppo.py
"""
from __future__ import annotations

import numpy as np

from rl_project.agents import PPOConfig
from rl_project.training import train_ppo
from rl_project.training.baselines import (
    DoNothingPolicy,
    NewsvendorPolicy,
    ReorderPointPolicy,
    evaluate_policy,
)
from rl_project.environments import InventoryEnv
from rl_project.environments.inventory_env import InventoryConfig


def main() -> None:
    cfg = PPOConfig(
        num_envs=8,
        rollout_length=128,
        epochs=10,
        minibatch_size=64,
        lr=3e-4,
        entropy_coef=0.0,
        clip_eps=0.2,
        normalize_advantages=True,
    )

    agent = train_ppo(
        "inventory",
        total_steps=200_000,
        eval_every=10_000,
        eval_episodes=10,
        action_mode="continuous",
        agent_config=cfg,
        log_dir="runs/ppo_inventory_tutorial",
        seed=42,
    )

    print("\n=== Held-out evaluation vs baselines ===")
    env = InventoryEnv(config=InventoryConfig(), action_mode="continuous")
    episodes = 50
    seed = 777

    class _RLPolicy:
        def act(self, env, obs):
            return agent.greedy_action(obs)

    policies = {
        "PPO (RL)": _RLPolicy(),
        "Do nothing": DoNothingPolicy(),
        "(s=20, S=60) reorder": ReorderPointPolicy(reorder_point=20, target_level=60),
        "Newsvendor": NewsvendorPolicy(),
    }

    print(f"{'Policy':<25} {'Profit':>12} {'±':>10} {'Stockouts':>12} {'Holding':>10}")
    print("-" * 75)
    results = {}
    for name, p in policies.items():
        r = evaluate_policy(p, env, episodes=episodes, seed=seed)
        results[name] = r
        print(
            f"{name:<25} {r['profit_mean']:>12.1f} {r['profit_std']:>10.1f} "
            f"{r['stockout_units_mean']:>12.1f} {r['holding_cost_mean']:>10.2f}",
        )

    best_baseline = max(
        [v["profit_mean"] for k, v in results.items() if k != "PPO (RL)"],
    )
    lift = (results["PPO (RL)"]["profit_mean"] - best_baseline) / abs(best_baseline) * 100
    print(f"\nPPO lift over best baseline: {lift:+.1f}%")


if __name__ == "__main__":
    main()
