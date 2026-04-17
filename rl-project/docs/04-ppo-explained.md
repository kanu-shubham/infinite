# 4. PPO — Proximal Policy Optimization

PPO (Schulman et al., 2017) is the default RL algorithm at OpenAI, DeepMind,
Anthropic, etc. It is an **on-policy, actor-critic, policy-gradient** method.

## 4.1 The Policy Gradient

We want to maximize expected return. The gradient is:

```
grad J(theta) = E [ grad log pi_theta(a|s) * A(s,a) ]
```

where `A(s,a) = Q(s,a) - V(s)` is the **advantage**: how much better action `a`
is than the average action in state `s`.

So the intuition is simple:
- If advantage is **positive**, **increase** the probability of that action.
- If advantage is **negative**, **decrease** it.

## 4.2 Why Plain Policy Gradient Breaks

Updating `theta` too aggressively can move the policy far from where it was,
destroying the value function estimate and collapsing training. We want
**small, safe updates** per batch.

## 4.3 The PPO Clipped Objective

Let `r(theta) = pi_theta(a|s) / pi_{theta_old}(a|s)` be the **probability ratio**.
Then PPO maximizes:

```
L^CLIP(theta) = E [ min( r(theta) * A , clip(r(theta), 1-eps, 1+eps) * A ) ]
```

- If `A > 0`: we want to increase `r`, but not beyond `1 + eps`. Clipping stops
  the gradient once we've gone far enough.
- If `A < 0`: we want to decrease `r`, but not below `1 - eps`.

Typical `eps = 0.2`.

The total loss combines three terms:

```
L = -L^CLIP + c1 * (V_theta(s) - V_target)^2 - c2 * entropy(pi_theta)
```

- Value loss: trains the critic.
- Entropy bonus: encourages exploration.

## 4.4 Generalized Advantage Estimation (GAE)

Advantages are computed with GAE for a good bias-variance tradeoff:

```
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
A_t     = delta_t + (gamma * lambda) * delta_{t+1} + (gamma * lambda)^2 * delta_{t+2} + ...
```

- `lambda = 1.0` -> Monte Carlo (low bias, high variance).
- `lambda = 0.0` -> TD(0) (high bias, low variance).
- `lambda = 0.95` is the usual sweet spot.

## 4.5 The Algorithm

```
Initialize actor pi_theta, critic V_phi
for iteration = 1, N:
    for t = 1, T:
        a ~ pi_theta(.|s)
        s', r, done = env.step(a)
        store (s, a, r, log_prob, v) in buffer
        s = s'
    compute advantages with GAE
    normalize advantages
    for K epochs:
        for minibatch in buffer:
            compute ratio r(theta)
            compute L^CLIP, L^VALUE, entropy
            loss = -L^CLIP + c1 * L^VALUE - c2 * entropy
            grad step on theta, phi
    clear buffer
```

Typical hyperparameters:

| Hyperparameter      | Value        |
| ------------------- | ------------ |
| rollout length T    | 128-2048     |
| epochs K            | 4-10         |
| minibatch size      | 64-256       |
| gamma               | 0.99         |
| lambda (GAE)        | 0.95         |
| clip epsilon        | 0.2          |
| learning rate       | 3e-4         |
| entropy coef c2     | 0.0-0.01     |
| value coef c1       | 0.5          |
| grad clip           | 0.5          |

## 4.6 Why PPO Works So Well in Practice

- **Safe updates** (clipping) — rarely diverges.
- **Works with discrete AND continuous** actions.
- **Actor-critic** reduces variance vs vanilla REINFORCE.
- **Simple** compared to TRPO (which uses a trust region).
- **Easy to parallelize** across environments.

## 4.7 Common Pitfalls

- Forgetting to **normalize advantages** per minibatch -> unstable.
- Forgetting to **zero the optimizer gradient**.
- Using **old log-probs** that were not detached.
- Wrong **advantage bootstrapping** at the end of rollout (must handle `done`).
- Too large a **learning rate** with small batches.
