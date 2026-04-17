"""Tutorial 2: Solve CartPole with our PPO implementation.

PPO collects `N` rollouts in parallel, computes GAE advantages, and does
multiple epochs of clipped-policy-gradient updates per rollout.

Run:
    python tutorials/02_cartpole_ppo.py
"""
from __future__ import annotations

import numpy as np

from rl_project.agents import PPOAgent, PPOConfig
from rl_project.training.envs import make_env, make_vec_env


def main() -> None:
    cfg = PPOConfig(
        num_envs=8,
        rollout_length=128,
        epochs=4,
        minibatch_size=64,
        lr=3e-4,
        entropy_coef=0.0,
        clip_eps=0.2,
    )
    envs = make_vec_env("cartpole", cfg.num_envs, seed=0)
    eval_env = make_env("cartpole", seed=999)

    obs_dim = int(np.prod(envs.single_observation_space.shape))
    n_actions = int(envs.single_action_space.n)

    agent = PPOAgent(obs_dim, n_actions, discrete=True, config=cfg, seed=0)

    obs, _ = envs.reset(seed=0)
    dones = np.zeros(cfg.num_envs, dtype=np.float32)
    ep_returns = np.zeros(cfg.num_envs, dtype=np.float32)
    history: list[float] = []

    total_steps = 80_000
    step = 0
    while step < total_steps:
        for _ in range(cfg.rollout_length):
            action, log_prob, value = agent.act(obs)
            next_obs, reward, term, trunc, _ = envs.step(action)
            done = np.logical_or(term, trunc).astype(np.float32)
            agent.buffer.add(obs, action, log_prob, reward.astype(np.float32), value, dones)
            ep_returns += reward
            for i, d in enumerate(done):
                if d:
                    history.append(float(ep_returns[i]))
                    ep_returns[i] = 0.0
            obs, dones = next_obs, done
            step += cfg.num_envs

        last_values = agent.value(obs)
        agent.buffer.compute_gae(last_values=last_values, last_dones=dones)
        stats = agent.update()

        avg50 = float(np.mean(history[-50:])) if history else 0.0
        eval_ret = _evaluate(agent, eval_env, 5)
        print(
            f"step={step:>6d}  train_ret(last50)={avg50:6.1f}  "
            f"eval_ret={eval_ret:6.1f}  kl={stats['approx_kl']:.3f}  "
            f"clip={stats['clip_frac']:.2f}",
        )
        if eval_ret >= 475.0:
            print("Solved CartPole-v1!")
            break

    envs.close()


def _evaluate(agent: PPOAgent, env, n: int) -> float:
    totals = []
    for ep in range(n):
        obs, _ = env.reset(seed=10_000 + ep)
        done, total = False, 0.0
        while not done:
            a = int(agent.greedy_action(obs))
            obs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            total += float(r)
        totals.append(total)
    return float(np.mean(totals))


if __name__ == "__main__":
    main()
