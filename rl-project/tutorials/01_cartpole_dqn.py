"""Tutorial 1: Solve CartPole with our DQN implementation.

CartPole is the "hello world" of RL. A cart moves left/right to balance a
pole. The observation is 4D (cart pos/vel, pole angle/vel) and the action is
a simple left/right.

CartPole-v1 is considered "solved" at an average return >= 475 over 100
episodes. Our vanilla DQN usually gets there in a few minutes of CPU time.

Run:
    python tutorials/01_cartpole_dqn.py
"""
from __future__ import annotations

import numpy as np

from rl_project.agents import DQNAgent, DQNConfig
from rl_project.training.envs import make_env


def main() -> None:
    env = make_env("cartpole", seed=0)
    eval_env = make_env("cartpole", seed=123)

    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)

    # CartPole is small; a tiny net and aggressive epsilon decay suffice.
    agent = DQNAgent(
        obs_dim,
        n_actions,
        config=DQNConfig(
            lr=5e-4,
            buffer_size=50_000,
            min_buffer=1_000,
            epsilon_decay_steps=10_000,
            hidden=(64, 64),
        ),
        seed=0,
    )

    obs, _ = env.reset(seed=0)
    ep_return = 0.0
    ep_returns: list[float] = []

    for step in range(1, 30_001):
        a = agent.select_action(obs)
        next_obs, r, term, trunc, _ = env.step(a)
        agent.observe(obs, a, float(r), next_obs, bool(term))
        ep_return += float(r)
        obs = next_obs

        if term or trunc:
            ep_returns.append(ep_return)
            obs, _ = env.reset()
            ep_return = 0.0

        agent.update()

        if step % 2_000 == 0:
            avg50 = float(np.mean(ep_returns[-50:])) if ep_returns else 0.0
            eval_ret = _evaluate(agent, eval_env, 5)
            print(
                f"step={step:>6d}  train_ret(last50)={avg50:6.1f}  "
                f"eval_ret={eval_ret:6.1f}  eps={agent.epsilon():.2f}",
            )
            if eval_ret >= 475.0:
                print("Solved CartPole-v1!")
                return


def _evaluate(agent: DQNAgent, env, n: int) -> float:
    totals = []
    for ep in range(n):
        obs, _ = env.reset(seed=100_000 + ep)
        done, total = False, 0.0
        while not done:
            a = agent.select_action(obs, greedy=True)
            obs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            total += float(r)
        totals.append(total)
    return float(np.mean(totals))


if __name__ == "__main__":
    main()
