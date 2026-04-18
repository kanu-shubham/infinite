"""
Project 29: Reinforcement Learning
=====================================
RL trains agents to make decisions by rewarding good behavior.
It powers AlphaGo, ChatGPT's RLHF, and game-playing AIs.

What you'll learn:
- The RL framework: agent, environment, state, action, reward
- Q-Learning: tabular RL for small state spaces
- Deep Q-Network (DQN): Q-learning with a neural network
- Policy Gradient (REINFORCE): learning directly from trajectories
- Key concepts: exploration vs exploitation, replay buffer, target network
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random

# Try to import gymnasium (modern OpenAI Gym)
try:
    import gymnasium as gym
    GYM_AVAILABLE = True
    GYM_NAME = "gymnasium"
except ImportError:
    try:
        import gym
        GYM_AVAILABLE = True
        GYM_NAME = "gym"
    except ImportError:
        GYM_AVAILABLE = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ════════════════════════════════════════════════════════════════════════════
#  PART 1: Q-Learning (tabular, no neural network)
# ════════════════════════════════════════════════════════════════════════════

class GridWorld:
    """
    Simple 5x5 grid world:
    - S: start (0,0)
    - G: goal (4,4) — reward +10
    - H: hole (2,2) — reward -10, episode ends
    - Every step: reward -1 (incentivize shortest path)
    """
    ACTIONS = [(0,1), (0,-1), (1,0), (-1,0)]   # right, left, down, up
    ACTION_NAMES = ["→", "←", "↓", "↑"]

    def __init__(self, size=5):
        self.size = size
        self.goal = (size-1, size-1)
        self.hole = (size//2, size//2)
        self.reset()

    def reset(self):
        self.pos = (0, 0)
        return self.pos

    def step(self, action):
        dr, dc = self.ACTIONS[action]
        r, c   = self.pos[0] + dr, self.pos[1] + dc
        r = max(0, min(self.size-1, r))
        c = max(0, min(self.size-1, c))
        self.pos = (r, c)

        if self.pos == self.goal:
            return self.pos, +10, True
        if self.pos == self.hole:
            return self.pos, -10, True
        return self.pos, -1, False


def q_learning(n_episodes=2000, alpha=0.1, gamma=0.99,
               epsilon_start=1.0, epsilon_end=0.01):
    """
    Tabular Q-Learning.

    Q-table: Q[state, action] = expected future reward
    Update rule (Bellman equation):
      Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]
                              ↑         ↑
                           reward    best future value

    ε-greedy exploration:
      With prob ε: random action  (explore)
      With prob 1-ε: argmax Q     (exploit)
    """
    env = GridWorld()
    n_states  = env.size * env.size
    n_actions = len(env.ACTIONS)

    Q = np.zeros((env.size, env.size, n_actions))   # Q[row, col, action]

    episode_rewards = []
    epsilon = epsilon_start

    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0
        epsilon = max(epsilon_end, epsilon_start * (1 - ep / n_episodes))

        for step in range(100):
            # ε-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.randint(n_actions)     # explore
            else:
                action = Q[state].argmax()                 # exploit

            next_state, reward, done = env.step(action)

            # Q-learning update (Bellman equation)
            best_next = Q[next_state].max()
            Q[state][action] += alpha * (reward + gamma * best_next - Q[state][action])

            state = next_state
            total_reward += reward
            if done:
                break

        episode_rewards.append(total_reward)

    return Q, episode_rewards, env


def visualize_q_table(Q, env):
    """Show which action the agent prefers in each cell."""
    arrows = {"→": (0.3, 0), "←": (-0.3, 0), "↓": (0, -0.3), "↑": (0, 0.3)}
    action_symbols = GridWorld.ACTION_NAMES

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.5, env.size - 0.5)
    ax.set_ylim(-0.5, env.size - 0.5)
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.grid(True, linewidth=2)

    for r in range(env.size):
        for c in range(env.size):
            best_action = Q[r, c].argmax()
            symbol = action_symbols[best_action]
            dx, dy = arrows[symbol]
            ax.annotate("", xy=(c+dx, env.size-1-r+dy), xytext=(c, env.size-1-r),
                        arrowprops=dict(arrowstyle="->", color="blue", lw=2))
            q_val = Q[r, c].max()
            ax.text(c - 0.35, env.size-1-r - 0.35, f"{q_val:.1f}", fontsize=7, color="gray")

    # Highlight special cells
    for (r, c), color, label in [
        ((0, 0), "green",  "S"),
        (env.goal, "gold",   "G"),
        (env.hole, "red",    "H"),
    ]:
        rect = plt.Rectangle((c-0.5, env.size-1-r-0.5), 1, 1, color=color, alpha=0.3)
        ax.add_patch(rect)
        ax.text(c, env.size-1-r, label, ha="center", va="center", fontsize=14, fontweight="bold")

    ax.set_title("Q-Learning: Optimal Policy\n"
                 "(arrows show best action, numbers show Q-value)")
    plt.tight_layout()
    plt.savefig("29_q_learning_policy.png", dpi=100)
    print("Saved Q-table policy to 29_q_learning_policy.png")


# ════════════════════════════════════════════════════════════════════════════
#  PART 2: Deep Q-Network (DQN)
# ════════════════════════════════════════════════════════════════════════════

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    """
    Experience Replay Buffer.

    Why replay?
    - Breaks temporal correlations (consecutive steps are highly correlated)
    - Reuses each experience multiple times (sample efficiency)
    - Enables mini-batch training
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    """
    Deep Q-Network: approximates Q(s, a) for all actions simultaneously.

    Input:  state vector
    Output: Q-value for each action (one output per action)

    This replaces the Q-table with a neural network.
    Q-table: can only handle discrete, small state spaces
    DQN:     handles continuous, high-dimensional states (pixels!)
    """
    def __init__(self, state_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """
    DQN Agent with two key improvements over naive DQN:
    1. Experience Replay: store and randomly sample past transitions
    2. Target Network: separate network for stable Q targets (updated slowly)
    """
    def __init__(self, state_dim, n_actions, lr=1e-3, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.n_actions = n_actions
        self.gamma     = gamma
        self.epsilon   = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Online network (trained every step)
        self.policy_net = DQN(state_dim, n_actions).to(DEVICE)
        # Target network (copied from policy_net every N steps — stable targets)
        self.target_net = DQN(state_dim, n_actions).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory    = ReplayBuffer(capacity=10000)
        self.steps     = 0

    def select_action(self, state):
        """ε-greedy: explore randomly or exploit learned Q-values."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            return self.policy_net(state_t).argmax().item()

    def store(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self, batch_size=64):
        if len(self.memory) < batch_size:
            return None

        batch = self.memory.sample(batch_size)
        states      = torch.tensor([t.state      for t in batch], dtype=torch.float32).to(DEVICE)
        actions     = torch.tensor([t.action     for t in batch], dtype=torch.long).to(DEVICE)
        rewards     = torch.tensor([t.reward     for t in batch], dtype=torch.float32).to(DEVICE)
        next_states = torch.tensor([t.next_state for t in batch], dtype=torch.float32).to(DEVICE)
        dones       = torch.tensor([t.done       for t in batch], dtype=torch.float32).to(DEVICE)

        # Current Q values: Q(s, a) for taken actions
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values: r + γ max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = F.smooth_l1_loss(current_q, target_q)   # Huber loss: robust to outliers
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update target network every 100 steps
        self.steps += 1
        if self.steps % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss.item()


def train_dqn_gridworld(n_episodes=500):
    """DQN on the GridWorld — demonstrates DQN on a simple env."""
    print("\n── DQN on GridWorld ──\n")

    env = GridWorld(size=5)
    state_dim = 2    # (row, col)
    n_actions = 4

    agent = DQNAgent(state_dim, n_actions, lr=5e-4)
    rewards_history = []

    for ep in range(1, n_episodes + 1):
        state = env.reset()
        state_vec = list(state)
        total_reward = 0

        for _ in range(100):
            action = agent.select_action(state_vec)
            next_state, reward, done = env.step(action)
            next_vec = list(next_state)

            agent.store(state_vec, action, reward, next_vec, done)
            agent.train_step(batch_size=32)

            state_vec = next_vec
            total_reward += reward
            if done:
                break

        rewards_history.append(total_reward)

        if ep % 100 == 0:
            avg = np.mean(rewards_history[-50:])
            print(f"  Episode {ep:4d}/{n_episodes} | Avg Reward (last 50): {avg:.2f} | ε: {agent.epsilon:.3f}")

    return agent, rewards_history


# ════════════════════════════════════════════════════════════════════════════
#  PART 3: Policy Gradient (REINFORCE)
# ════════════════════════════════════════════════════════════════════════════

class PolicyNetwork(nn.Module):
    """
    Policy network π_θ(a|s): maps state → action probabilities.

    Unlike DQN (value-based), policy gradient directly optimizes the policy.
    Works better for continuous action spaces and stochastic policies.
    """
    def __init__(self, state_dim, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)


def reinforce(n_episodes=1000, gamma=0.99, lr=1e-3):
    """
    REINFORCE (Williams, 1992) — simplest policy gradient algorithm.

    For each episode:
    1. Run policy π_θ, collect (s_t, a_t, r_t) trajectory
    2. Compute discounted returns G_t = Σ γ^k r_{t+k}
    3. Update: θ ← θ + α Σ_t G_t ∇log π_θ(a_t|s_t)

    Intuition: increase probability of actions that led to high returns,
               decrease probability of actions that led to low returns.
    """
    print("\n── REINFORCE (Policy Gradient) on GridWorld ──\n")

    env = GridWorld(size=5)
    policy = PolicyNetwork(state_dim=2, n_actions=4, hidden=64).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    rewards_history = []

    for ep in range(1, n_episodes + 1):
        # Collect trajectory
        states, actions, rewards = [], [], []
        state = env.reset()

        for _ in range(50):
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            probs   = policy(state_t)
            dist    = torch.distributions.Categorical(probs)
            action  = dist.sample().item()

            next_state, reward, done = env.step(action)
            states.append(state_t)
            actions.append(action)
            rewards.append(reward)
            state = next_state
            if done:
                break

        total_reward = sum(rewards)
        rewards_history.append(total_reward)

        # Compute discounted returns
        G, returns = 0.0, []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns_t = torch.tensor(returns, dtype=torch.float32).to(DEVICE)

        # Normalize returns (reduce variance)
        if len(returns) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # Policy gradient update
        loss = 0.0
        for state_t, action, G_t in zip(states, actions, returns_t):
            probs  = policy(state_t)
            dist   = torch.distributions.Categorical(probs)
            log_p  = dist.log_prob(torch.tensor(action).to(DEVICE))
            loss  -= log_p * G_t   # negative because we maximize

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        if ep % 200 == 0:
            avg = np.mean(rewards_history[-100:])
            print(f"  Episode {ep:4d}/{n_episodes} | Avg Reward (last 100): {avg:.2f}")

    return policy, rewards_history


def main():
    print("=== Reinforcement Learning ===")
    print(f"Device: {DEVICE}\n")

    print("── The RL Framework ──\n")
    print("""
  Agent ←→ Environment

  At each step t:
    Agent observes state s_t
    Agent selects action a_t
    Environment returns reward r_t and next state s_{t+1}
    Agent updates its policy

  Goal: maximize cumulative discounted reward: Σ γ^t r_t

  Key concepts:
    Policy π(a|s):    probability of taking action a in state s
    Value V(s):       expected future reward from state s
    Q-value Q(s,a):   expected future reward from state s, taking action a
    Bellman equation: Q(s,a) = r + γ max_a' Q(s', a')
    """)

    # ── Q-Learning ────────────────────────────────────────────────────────
    print("── Part 1: Q-Learning (Tabular) ──\n")
    Q, q_rewards, env = q_learning(n_episodes=2000)
    visualize_q_table(Q, env)

    # ── DQN ──────────────────────────────────────────────────────────────
    agent, dqn_rewards = train_dqn_gridworld(n_episodes=500)

    # ── Policy Gradient ──────────────────────────────────────────────────
    policy, pg_rewards = reinforce(n_episodes=1000)

    # ── Plot all reward curves ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    def smooth(x, w=50):
        return np.convolve(x, np.ones(w)/w, mode="valid")

    axes[0].plot(q_rewards, alpha=0.3, color="blue")
    axes[0].plot(smooth(q_rewards), color="blue", linewidth=2, label="Q-Learning")
    axes[0].set_title("Q-Learning (Tabular)\nEpisode Reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(dqn_rewards, alpha=0.3, color="green")
    axes[1].plot(smooth(dqn_rewards, w=30), color="green", linewidth=2, label="DQN")
    axes[1].set_title("DQN (Deep Q-Network)\nEpisode Reward")
    axes[1].set_xlabel("Episode")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(pg_rewards, alpha=0.3, color="red")
    axes[2].plot(smooth(pg_rewards, w=100), color="red", linewidth=2, label="REINFORCE")
    axes[2].set_title("REINFORCE (Policy Gradient)\nEpisode Reward")
    axes[2].set_xlabel("Episode")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Reinforcement Learning Algorithms Comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig("29_rl_comparison.png", dpi=100)
    print("\nSaved RL comparison to 29_rl_comparison.png")

    # ── Algorithm comparison ──────────────────────────────────────────────
    print("\n── RL Algorithm Comparison ──\n")
    print(f"{'Algorithm':<20} {'Type':<20} {'State Space':<20} {'When to use'}")
    print("-" * 80)
    rows = [
        ("Q-Learning",     "Value-based",    "Small, discrete",   "Simple tabular problems"),
        ("DQN",            "Value-based",    "Large/continuous",  "Atari games, discrete actions"),
        ("REINFORCE",      "Policy gradient","Any",               "When you need stochastic policies"),
        ("A3C/A2C",        "Actor-Critic",   "Any",               "Faster policy gradient training"),
        ("PPO",            "Policy gradient","Any",               "Industry standard (ChatGPT RLHF)"),
        ("SAC",            "Actor-Critic",   "Continuous",        "Robotics, continuous control"),
    ]
    for row in rows:
        print(f"{row[0]:<20} {row[1]:<20} {row[2]:<20} {row[3]}")

    if not GYM_AVAILABLE:
        print("\nTo use real environments (CartPole, Atari, etc.):")
        print("  pip install gymnasium")
        print("  pip install gymnasium[atari]")
        print("\nThen replace GridWorld with:")
        print("  env = gymnasium.make('CartPole-v1')")

    print("\n=== Summary ===")
    print("Key concepts mastered:")
    print("  ✓ RL framework — agent, state, action, reward, policy")
    print("  ✓ Bellman equation — recursive definition of Q-values")
    print("  ✓ ε-greedy — exploration vs exploitation tradeoff")
    print("  ✓ Q-Learning — tabular value function update")
    print("  ✓ DQN — Q-learning with neural network approximation")
    print("  ✓ Replay buffer — breaks temporal correlations")
    print("  ✓ Target network — stable training targets")
    print("  ✓ REINFORCE — policy gradient, log-probability trick")
    print("  ✓ Return normalization — reduces gradient variance")


if __name__ == "__main__":
    main()
