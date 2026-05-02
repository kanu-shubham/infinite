"""
Stage 2 — Reward Model (RM).

Humans rank pairs: (prompt, response_w, response_l) where w is preferred.
Under the Bradley-Terry assumption,
    P(w > l | c) = σ(r(c, w) - r(c, l)).
We fit r_φ by minimizing -log σ(r_φ(c, w) - r_φ(c, l)). Note r is identifiable
only up to a per-context constant; that constant cancels in the loss.

Output: alignment/rm.json — the learned reward used by PPO and as a teacher
in some distillation setups.
"""
import json
import random

from common import (
    ACTIONS, CONTEXTS, N_ACT, N_CTX, TRUE_REWARD,
    sample_preferences, sigmoid,
)


def train_reward_model(n_pairs=3000, epochs=80, lr=0.1, seed=0):
    random.seed(seed)
    R = [[0.0] * N_ACT for _ in range(N_CTX)]
    data = sample_preferences(n=n_pairs, seed=seed)

    for ep in range(epochs):
        random.shuffle(data)
        for c, aw, al in data:
            diff = R[c][aw] - R[c][al]
            p = sigmoid(diff)
            # L = -log σ(diff). dL/d(diff) = p - 1.
            d = p - 1.0
            R[c][aw] -= lr * d
            R[c][al] -= lr * (-d)
    return R


def report(R):
    correct = 0
    for c in range(N_CTX):
        true_best = max(range(N_ACT), key=lambda a: TRUE_REWARD[c][a])
        rm_best = max(range(N_ACT), key=lambda a: R[c][a])
        ok = "OK" if true_best == rm_best else "MISS"
        correct += int(true_best == rm_best)
        print(f"  {CONTEXTS[c]:>10}: true={ACTIONS[true_best]:<10} "
              f"rm={ACTIONS[rm_best]:<10} [{ok}]")
    print(f"  argmax-agreement: {correct}/{N_CTX}")


if __name__ == "__main__":
    R = train_reward_model()
    print("[RM] learned reward model:")
    report(R)
    with open("alignment/rm.json", "w") as f:
        json.dump(R, f)
    print("saved -> alignment/rm.json")
