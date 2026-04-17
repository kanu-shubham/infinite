"""DQN training loop."""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from rl_project.agents import DQNAgent, DQNConfig
from rl_project.training.envs import make_env
from rl_project.utils import Logger


def train_dqn(
    env_name: str,
    *,
    total_steps: int = 50_000,
    eval_every: int = 5_000,
    eval_episodes: int = 10,
    action_mode: str = "discrete",
    horizon: int | None = None,
    agent_config: DQNConfig | None = None,
    log_dir: str = "runs/dqn",
    seed: int = 0,
    device: str | None = None,
) -> DQNAgent:
    env = make_env(env_name, action_mode=action_mode, horizon=horizon, seed=seed)
    eval_env = make_env(env_name, action_mode=action_mode, horizon=horizon, seed=seed + 1000)

    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)

    agent = DQNAgent(obs_dim, n_actions, config=agent_config, device=device, seed=seed)
    logger = Logger(log_dir, name="train")

    obs, _ = env.reset(seed=seed)
    ep_return = 0.0
    ep_len = 0
    returns_window: deque[float] = deque(maxlen=50)
    best_eval = -np.inf

    for step in range(1, total_steps + 1):
        a = agent.select_action(obs)
        next_obs, r, term, trunc, _ = env.step(a)
        # Bellman target should only zero bootstrap on *true* terminations.
        done_bootstrap = bool(term)
        agent.observe(obs, a, float(r), next_obs, done_bootstrap)
        ep_return += float(r)
        ep_len += 1
        obs = next_obs

        if term or trunc:
            returns_window.append(ep_return)
            obs, _ = env.reset()
            ep_return, ep_len = 0.0, 0

        metrics = agent.update()
        if step % eval_every == 0:
            eval_ret = _evaluate(agent, eval_env, eval_episodes)
            logger.log(
                step,
                train_return=float(np.mean(returns_window)) if returns_window else 0.0,
                eval_return=eval_ret,
                epsilon=agent.epsilon(),
                loss=(metrics or {}).get("loss", 0.0),
            )
            if eval_ret > best_eval:
                best_eval = eval_ret
                agent.save(Path(log_dir) / "best.pt")

    agent.save(Path(log_dir) / "final.pt")
    logger.close()
    return agent


def _evaluate(agent: DQNAgent, env, episodes: int) -> float:
    totals: list[float] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=1_000_000 + ep)
        done = False
        total = 0.0
        while not done:
            a = agent.select_action(obs, greedy=True)
            obs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            total += float(r)
        totals.append(total)
    return float(np.mean(totals))
