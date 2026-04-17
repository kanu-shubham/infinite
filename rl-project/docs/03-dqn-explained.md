# 3. DQN — Deep Q-Networks

## 3.1 The Idea in One Paragraph

Q-Learning says: if I know `Q*(s,a)` (the optimal action value), I should just
pick `argmax_a Q*(s,a)`. DQN estimates `Q*(s,a)` with a **neural network**
`Q_theta(s,a)` and trains it to satisfy the **Bellman optimality equation**:

```
Q*(s,a) = E[ r + gamma * max_{a'} Q*(s', a') ]
```

## 3.2 The Loss

For each transition `(s, a, r, s', done)`:

```
y = r + gamma * max_{a'} Q_theta_target(s', a')    if not done else r
L = ( Q_theta(s, a) - y )^2                         # MSE or Huber
```

`Q_theta_target` is a **separate, slowly updated copy** of the network — this
stops the target from moving every step and is crucial for stability.

## 3.3 The Two Key Tricks (from the 2015 Nature paper)

1. **Experience Replay Buffer**
   - Store transitions `(s, a, r, s', done)` in a FIFO buffer (typically 100k-1M).
   - Sample **random minibatches** to train — breaks temporal correlation.
   - Enables **off-policy** learning: old data is still useful.

2. **Target Network**
   - A copy of Q that is only updated every `N` steps (e.g. 1000).
   - Or soft-updated with `tau` (e.g. `0.005`):
     `theta_target <- tau * theta + (1 - tau) * theta_target`.
   - Without this, the target changes every gradient step -> divergence.

## 3.4 The Algorithm

```
Initialize Q_theta, Q_target <- Q_theta, empty replay buffer D
for episode = 1, M:
    s = env.reset()
    while not done:
        with prob epsilon: a = random action
        else: a = argmax Q_theta(s, .)
        s', r, done = env.step(a)
        D.add((s, a, r, s', done))

        if len(D) > batch_size:
            sample batch from D
            y = r + gamma * max_a' Q_target(s', a') * (1 - done)
            loss = (Q_theta(s, a) - y)^2
            backprop, step optimizer

        if step % target_update_freq == 0:
            Q_target <- Q_theta
        s = s'
    decay epsilon
```

## 3.5 Improvements (briefly)

These are common "DQN extensions" you should know by name:

| Name              | What it adds |
| ----------------- | ------------ |
| **Double DQN**    | Uses online net to select action, target net to evaluate it — reduces overestimation bias. |
| **Dueling DQN**   | Splits Q into V(s) + A(s,a); better when many actions have similar value. |
| **Prioritized ER**| Samples transitions with higher TD error more often. |
| **Noisy Nets**    | Replaces epsilon-greedy with noisy weights. |
| **Rainbow**       | Combines all of the above. |

Our implementation uses vanilla DQN + Double DQN + soft target updates, which
is a sweet spot of simplicity and quality.

## 3.6 When to Use DQN

- **Discrete action space** (continuous needs DDPG / SAC / PPO).
- When you can generate many samples (off-policy lets you reuse data).
- When a single-step bootstrap is reasonable (not very long-horizon).

## 3.7 Common Pitfalls

- **Forgetting to set `done=True` only on real terminal**, not on time-limit
  truncation (it breaks the Bellman equation).
- **Huge Q-values** exploding — use **Huber loss** and **reward clipping**.
- **Too small replay buffer** — agent forgets.
- **Epsilon decays too fast** — stuck in local optimum.
