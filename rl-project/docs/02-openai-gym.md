# 2. OpenAI Gym (Gymnasium) — The Standard RL API

[`gymnasium`](https://gymnasium.farama.org/) (the maintained fork of OpenAI Gym)
is the de-facto interface for RL environments. Every environment implements the
same five things, which is why we can swap CartPole with our inventory env
without changing the agent code.

## 2.1 The API

```python
import gymnasium as gym

env = gym.make("CartPole-v1")

observation, info = env.reset(seed=42)     # start a new episode

for t in range(500):
    action = env.action_space.sample()     # random action (for demo)
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

Just five things to understand:

| Method / attribute | Purpose |
| ------------------ | ------- |
| `env.reset()`      | Start a new episode. Returns `(obs, info)`. |
| `env.step(a)`      | Apply action `a`. Returns `(obs, reward, terminated, truncated, info)`. |
| `env.observation_space` | Shape / bounds of observations (`Box`, `Discrete`, ...). |
| `env.action_space`      | Shape / bounds of actions. |
| `terminated`       | Episode ended for a "real" reason (goal reached, failed). |
| `truncated`        | Episode ended due to a time limit. |

## 2.2 Spaces

- `Box(low, high, shape)` — continuous (e.g. `[-1.0, 1.0]`).
- `Discrete(n)` — integer in `{0, ..., n-1}`.
- `MultiDiscrete`, `MultiBinary`, `Dict`, `Tuple` — composites.

CartPole: `obs = Box(4,)` (cart pos, cart vel, pole angle, pole vel),
`action = Discrete(2)` (left or right).

## 2.3 Writing a Custom Environment

Subclass `gym.Env` and implement:

```python
class MyEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, ...):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=..., high=..., shape=(n,))
        self.action_space = gym.spaces.Discrete(k)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # initialize state
        return observation, info

    def step(self, action):
        # apply action, compute reward, check terminal
        return observation, reward, terminated, truncated, info
```

See [`src/rl_project/environments/inventory_env.py`](../src/rl_project/environments/inventory_env.py)
for a full production example.

## 2.4 Tips

- **Always `seed` your env** for reproducibility.
- **Normalize observations** (zero mean, unit variance) — huge effect on
  training stability.
- **Clip rewards** during training if they are unbounded (e.g. finance).
- **Vectorize envs** (`gym.vector.SyncVectorEnv`) to parallelize rollouts — PPO
  in particular benefits a lot.
- Use `gymnasium.wrappers` (e.g. `TimeLimit`, `RecordEpisodeStatistics`,
  `NormalizeObservation`) — don't reinvent.
