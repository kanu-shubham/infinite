"""
Stage 3 — RLHF with PPO.

Reward used by RL:
    R(c, a) = r_φ(c, a) - β * (log π_θ(a|c) - log π_ref(a|c))
The KL penalty pulls π_θ toward π_ref so the policy doesn't drift to
adversarial high-reward but degenerate outputs ("reward hacking").

PPO clipped surrogate (per sample, with advantage A and importance ratio
ρ = π_new / π_old):
    L = E[ min( ρ * A, clip(ρ, 1-ε, 1+ε) * A ) ]
We MAXIMIZE L. The clip stops too-large policy updates per step.

Notes vs. real LLM RLHF:
    * Sequence-level vs token-level: in real LLM PPO the KL penalty is added
      per-token and reward is given at end-of-sequence; here it's a 1-step
      bandit so they collapse.
    * We use a running-mean baseline instead of a learned value head.

WATCH FOR REWARD HACKING. PPO maximizes the LEARNED reward r_φ, not the
true reward r*. If your RM is even slightly miscalibrated (e.g. it ranks
"hi" above "hello!" for the greet context when r* says the opposite),
PPO will faithfully push the policy onto r_φ's argmax and your true-reward
score can DROP relative to SFT. The KL penalty slows but does not prevent
this. This is the central RLHF failure mode and is a key reason DPO is
attractive: by skipping the explicit RM, DPO optimizes against the
preference data directly and removes one place where errors compound.
"""
import json
import math
import random

from common import (
    N_ACT, N_CTX, Policy, expected_true_reward, grad_logp, show_policy,
    zero_grid,
)


def train_ppo(seed=0):
    random.seed(seed)
    sft_logits = json.load(open("alignment/sft_policy.json"))
    rm = json.load(open("alignment/rm.json"))

    pol = Policy(sft_logits)
    ref = Policy(sft_logits)  # frozen reference

    beta = 0.05
    eps_clip = 0.2
    n_iter = 80
    batch = 64
    ppo_epochs = 4
    lr = 0.02

    print(f"[PPO] start  E[r*] = {expected_true_reward(pol):+.3f}")
    for it in range(n_iter):
        # ---- collect rollouts under the current ("old") policy ----
        rollouts = []
        for _ in range(batch):
            c = random.randrange(N_CTX)
            a = pol.sample(c)
            r_phi = rm[c][a]
            old_logp = pol.logp(c, a)
            ref_logp = ref.logp(c, a)
            shaped = r_phi - beta * (old_logp - ref_logp)
            rollouts.append((c, a, old_logp, shaped))

        baseline = sum(x[3] for x in rollouts) / len(rollouts)
        advs = [x[3] - baseline for x in rollouts]

        # ---- multiple PPO epochs over the same rollouts ----
        for _ in range(ppo_epochs):
            grad = zero_grid()
            for (c, a, old_logp, _), adv in zip(rollouts, advs):
                ps = pol.probs(c)
                new_logp = math.log(ps[a] + 1e-30)
                ratio = math.exp(new_logp - old_logp)

                # If we're outside the trust region in the wrong direction,
                # the surrogate gradient is exactly zero (the clipped term flat).
                if adv >= 0 and ratio > 1 + eps_clip:
                    continue
                if adv < 0 and ratio < 1 - eps_clip:
                    continue

                # ∇(ratio * A) = A * ratio * ∇log π
                glp = grad_logp(c, a, ps)
                coef = adv * ratio
                for cc in range(N_CTX):
                    for aa in range(N_ACT):
                        grad[cc][aa] += coef * glp[cc][aa]

            n = len(rollouts)
            for cc in range(N_CTX):
                for aa in range(N_ACT):
                    grad[cc][aa] = -grad[cc][aa] / n  # negate to MAXIMIZE
            pol.adam_step(grad, lr=lr)

    print(f"[PPO] done   E[r*] = {expected_true_reward(pol):+.3f}")
    return pol


if __name__ == "__main__":
    pol = train_ppo()
    show_policy(pol, "PPO policy")
    with open("alignment/ppo_policy.json", "w") as f:
        json.dump(pol.logits, f)
    print("saved -> alignment/ppo_policy.json")
