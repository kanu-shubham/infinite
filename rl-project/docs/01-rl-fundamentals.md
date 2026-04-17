# 1. Reinforcement Learning Fundamentals

> Everything you need to know before touching code.

## 1.1 The Core Idea

Supervised learning needs **labeled data** (`x -> y`). Reinforcement learning
does not. Instead, an **agent interacts with an environment** and learns by
**trial and error**, guided by a **reward signal**.

Think of how you learned to ride a bicycle:

- You **observed** (saw the road, felt balance).
- You **acted** (pedaled, steered).
- You got a **reward** (stayed up) or **penalty** (fell down).
- Over time your **policy** (brain -> muscles mapping) improved.

## 1.2 The Loop

```
      action a_t
  +-------------->+
  |               |
Agent         Environment
  |               |
  +<--------------+
  state s_{t+1}, reward r_{t+1}
```

Each time step `t`:

1. Agent observes state `s_t`.
2. Agent picks action `a_t` using its policy `pi(a|s)`.
3. Environment transitions to `s_{t+1}` and emits reward `r_{t+1}`.
4. Repeat.

## 1.3 The Markov Decision Process (MDP)

An MDP is the mathematical framing of the above loop. It is a tuple
`(S, A, P, R, gamma)`:

- `S` — set of states.
- `A` — set of actions.
- `P(s'|s,a)` — transition probability.
- `R(s,a)` — reward function.
- `gamma` in `[0,1)` — **discount factor**. Future rewards are worth less than
  immediate rewards. `gamma = 0.99` is typical.

The **Markov property**: the future depends only on the current state, not the
history. If your problem violates this, include more info in the state
(e.g. last 4 frames in Atari).

## 1.4 Return, Value, Q-value

- **Return** `G_t = r_{t+1} + gamma*r_{t+2} + gamma^2*r_{t+3} + ...`
- **State value** `V^pi(s) = E[G_t | s_t = s]` under policy `pi`.
- **Action value** `Q^pi(s,a) = E[G_t | s_t=s, a_t=a]`.

Intuition: `V(s)` = "how good is this state" and `Q(s,a)` = "how good is taking
this action in this state".

## 1.5 The Bellman Equations

`V^pi(s) = E_a~pi, s'~P [ R(s,a) + gamma * V^pi(s') ]`

`Q^pi(s,a) = E_{s'~P} [ R(s,a) + gamma * E_{a'~pi} Q^pi(s',a') ]`

These recursive equations are the backbone of **value-based** methods like DQN.

## 1.6 Two Broad Families of Algorithms

| Family               | Learns         | Example |
| -------------------- | -------------- | ------- |
| **Value-based**      | Q(s,a)         | DQN     |
| **Policy-based**     | pi(a|s) directly | REINFORCE, PPO |
| **Actor-Critic**     | Both           | A2C, PPO |

- **DQN** (Deep Q-Network) estimates `Q(s,a)` with a neural network, then picks
  the action with highest Q. Works for **discrete action spaces**.
- **PPO** (Proximal Policy Optimization) learns `pi(a|s)` directly and uses a
  value function as a **critic** to reduce variance. Works for **discrete and
  continuous** actions and is the default for most production RL today.

## 1.7 Exploration vs Exploitation

If the agent always picks the best known action (greedy), it may miss a better
one. It must **explore**. Common tricks:

- **Epsilon-greedy** (DQN): with probability `epsilon` pick a random action.
- **Stochastic policy** (PPO): sample from `pi(a|s)` with entropy bonus.

## 1.8 On-policy vs Off-policy

- **On-policy** (PPO): learns from data collected by the **current** policy.
- **Off-policy** (DQN): learns from data collected by **any** policy, stored in
  a **replay buffer**. More sample-efficient but trickier to stabilize.

## 1.9 Common Terms Cheat Sheet

| Term        | Meaning |
| ----------- | ------- |
| Episode     | One run from reset to terminal state |
| Trajectory  | Sequence `s_0, a_0, r_1, s_1, ...` |
| Rollout     | A batch of trajectories used for one update |
| Horizon     | Max number of steps per episode |
| Return      | Sum of (discounted) rewards in an episode |
| Advantage   | `A(s,a) = Q(s,a) - V(s)`; how much better is action `a` than average |

## 1.10 What Makes RL Hard

- **Credit assignment**: if you win a chess game, which move caused the win?
- **Exploration**: you need to try bad things to find great things.
- **Non-stationarity**: as policy changes, the data distribution changes.
- **Sample inefficiency**: RL often needs millions of interactions.
- **Reward hacking**: agents will exploit any flaw in the reward function.

Knowing these pitfalls is half the battle in interviews.
