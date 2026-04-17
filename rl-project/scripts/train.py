"""CLI entry point for training DQN or PPO.

Example:
    python scripts/train.py --algo ppo --env inventory \
        --config src/rl_project/config/ppo_inventory.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from rl_project.agents import DQNConfig, PPOConfig
from rl_project.training import train_dqn, train_ppo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["dqn", "ppo"], required=True)
    parser.add_argument("--env", default="inventory")
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML config file (overrides defaults)")
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = {}
    if args.config is not None and args.config.exists():
        with args.config.open() as f:
            cfg = yaml.safe_load(f) or {}

    env_cfg = cfg.get("env", {})
    train_cfg = cfg.get("train", {})
    agent_cfg_dict = cfg.get("agent", {})

    env_name = args.env or env_cfg.get("name", "inventory")
    action_mode = env_cfg.get("action_mode", "discrete" if args.algo == "dqn" else "continuous")
    horizon = env_cfg.get("horizon")

    total_steps = args.total_steps or train_cfg.get("total_steps", 50_000)
    eval_every = train_cfg.get("eval_every", 5_000)
    eval_episodes = train_cfg.get("eval_episodes", 10)
    seed = args.seed if args.seed is not None else train_cfg.get("seed", 0)
    log_dir = str(args.log_dir or train_cfg.get("log_dir", f"runs/{args.algo}_{env_name}"))

    if args.algo == "dqn":
        agent_cfg = DQNConfig(**agent_cfg_dict) if agent_cfg_dict else None
        train_dqn(
            env_name,
            total_steps=total_steps,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            action_mode=action_mode,
            horizon=horizon,
            agent_config=agent_cfg,
            log_dir=log_dir,
            seed=seed,
            device=args.device,
        )
    else:
        agent_cfg = PPOConfig(**agent_cfg_dict) if agent_cfg_dict else None
        train_ppo(
            env_name,
            total_steps=total_steps,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            action_mode=action_mode,
            horizon=horizon,
            agent_config=agent_cfg,
            log_dir=log_dir,
            seed=seed,
            device=args.device,
        )


if __name__ == "__main__":
    main()
