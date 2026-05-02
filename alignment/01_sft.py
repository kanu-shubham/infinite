"""
Stage 1 — Supervised Fine-Tuning (SFT).

You have demonstrations (c, a*) from a "good" labeller. Maximize log π(a*|c).

This is just multi-class cross-entropy. It teaches *imitation* — the model
learns to copy the demonstrator. SFT alone cannot exceed the demonstrator
(no signal for "this is wrong"), so we follow it with preference learning.

Output: alignment/sft_policy.json (used by every downstream stage).
"""
import json
import random

from common import (
    N_CTX, N_ACT, TRUE_REWARD, Policy, expected_true_reward, grad_logp,
    show_policy, softmax, zero_grid,
)


def build_sft_dataset(n=400, expert_temp=2.0, seed=0):
    """Demonstrations sampled from softmax(temp * r*) — a good-but-noisy expert."""
    rng = random.Random(seed)
    data = []
    for _ in range(n):
        c = rng.randrange(N_CTX)
        ps = softmax([expert_temp * r for r in TRUE_REWARD[c]])
        u = rng.random()
        s = 0.0
        for a, p in enumerate(ps):
            s += p
            if u < s:
                data.append((c, a))
                break
    return data


def train_sft(epochs=30, lr=0.05, seed=0):
    random.seed(seed)
    pol = Policy()
    data = build_sft_dataset(seed=seed)

    print(f"[SFT] start  E[r*] = {expected_true_reward(pol):+.3f}")
    for ep in range(epochs):
        random.shuffle(data)
        for c, a in data:
            ps = pol.probs(c)
            # Loss = -log π(a|c). ∇loss = -∇log π.
            glp = grad_logp(c, a, ps)
            grad = zero_grid()
            for cc in range(N_CTX):
                for aa in range(N_ACT):
                    grad[cc][aa] = -glp[cc][aa]
            pol.adam_step(grad, lr=lr)
    print(f"[SFT] done   E[r*] = {expected_true_reward(pol):+.3f}")
    return pol


if __name__ == "__main__":
    pol = train_sft()
    show_policy(pol, "SFT policy")
    with open("alignment/sft_policy.json", "w") as f:
        json.dump(pol.logits, f)
    print("saved -> alignment/sft_policy.json")
