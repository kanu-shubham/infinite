"""PPO training loop."""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from rl_project.agents import PPOAgent, PPOConfig
from rl_project.training.envs import make_env, make_vec_env
from rl_project.utils import Logger


def train_ppo(
    env_name: str,
    *,
    total_steps: int = 200_000,
    eval_every: int = 10_000,
    eval_episodes: int = 10,
    action_mode: str = "continuous",
    horizon: int | None = None,
    agent_config: PPOConfig | None = None,
    log_dir: str = "runs/ppo",
    seed: int = 0,
    device: str | None = None,
) -> PPOAgent:
    cfg = agent_config or PPOConfig()
    envs = make_vec_env(
        env_name,
        cfg.num_envs,
        action_mode=action_mode,
        horizon=horizon,
        seed=seed,
    )
    eval_env = make_env(env_name, action_mode=action_mode, horizon=horizon, seed=seed + 10_000)

    obs_dim = int(np.prod(envs.single_observation_space.shape))
    discrete = hasattr(envs.single_action_space, "n")
    action_dim = (
        int(envs.single_action_space.n) if discrete
        else int(np.prod(envs.single_action_space.shape))
    )

    agent = PPOAgent(
        obs_dim, action_dim, discrete=discrete, config=cfg, device=device, seed=seed,
    )
    logger = Logger(log_dir, name="train")

    obs, _ = envs.reset(seed=seed)
    dones = np.zeros(cfg.num_envs, dtype=np.float32)
    ep_returns = np.zeros(cfg.num_envs, dtype=np.float32)
    returns_window: deque[float] = deque(maxlen=50)
    best_eval = -np.inf
    step = 0

    while step < total_steps:
        # --- Collect rollout of length T across N envs ---
        for _t in range(cfg.rollout_length):
            action, log_prob, value = agent.act(obs)
            # For continuous actions the env expects the raw array; for discrete we pass ints.
            step_action = action
            next_obs, reward, term, trunc, _ = envs.step(step_action)
            done = np.logical_or(term, trunc).astype(np.float32)
            agent.buffer.add(obs, action, log_prob, reward.astype(np.float32), value, dones)
            ep_returns += reward
            for i, d in enumerate(done):
                if d:
                    returns_window.append(float(ep_returns[i]))
                    ep_returns[i] = 0.0
            obs = next_obs
            dones = done
            step += cfg.num_envs

        # Bootstrap value at the end of rollout.
        last_values = agent.value(obs)
        agent.buffer.compute_gae(last_values=last_values, last_dones=dones)
        stats = agent.update()

        if step // cfg.rollout_length % max(1, eval_every // (cfg.rollout_length * cfg.num_envs)) == 0:
            eval_ret = _evaluate(agent, eval_env, eval_episodes)
            logger.log(
                step,
                train_return=float(np.mean(returns_window)) if returns_window else 0.0,
                eval_return=eval_ret,
                policy_loss=stats["policy_loss"],
                value_loss=stats["value_loss"],
                entropy=stats["entropy"],
                approx_kl=stats["approx_kl"],
                clip_frac=stats["clip_frac"],
            )
            if eval_ret > best_eval:
                best_eval = eval_ret
                agent.save(Path(log_dir) / "best.pt")

    agent.save(Path(log_dir) / "final.pt")
    logger.close()
    envs.close()
    return agent


def _evaluate(agent: PPOAgent, env, episodes: int) -> float:
    totals: list[float] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=1_000_000 + ep)
        done = False
        total = 0.0
        while not done:
            a = agent.greedy_action(obs)
            obs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            total += float(r)
        totals.append(total)
    return float(np.mean(totals))
