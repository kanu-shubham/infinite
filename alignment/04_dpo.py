"""
Stage 4 — Direct Preference Optimization (DPO).

KEY INSIGHT (Rafailov et al. 2023): under a Bradley-Terry preference model
and a KL-regularized RL objective, the OPTIMAL policy has the closed form
    π*(a|c) = (1/Z(c)) * π_ref(a|c) * exp(r(c,a)/β)
which can be inverted:
    r(c,a) = β * log(π*(a|c) / π_ref(a|c)) + β * log Z(c).
Substituting into the BT log-likelihood, log Z(c) cancels (it's the same
for both responses), and the preference loss becomes a function of the
*policy itself*:
    L_DPO = -E[ log σ( β * (
              log π_θ(a_w|c) - log π_θ(a_l|c)
            - log π_ref(a_w|c) + log π_ref(a_l|c)) ) ]

So we skip the RM AND the RL loop — one SFT-like training pass on
preference pairs is enough. That's why many shops dropped PPO:
    + No reward model to train, host, or de-bias.
    + No on-policy sampling loop, no value head, no KL controller, no
      PPO clip / advantage normalization to tune.
    + Stable; same infra as SFT.
    - Loses some flexibility: only learns from offline preference pairs;
      can't easily incorporate scalar / process rewards.
    - Sensitive to π_ref quality and to β; can over-fit on noisy prefs.
"""
import json
import math
import random

from common import (
    N_ACT, N_CTX, Policy, expected_true_reward, grad_logp, sample_preferences,
    show_policy, sigmoid, softmax, zero_grid,
)


def train_dpo(beta=0.5, epochs=40, lr=0.05, seed=0):
    random.seed(seed)
    sft_logits = json.load(open("alignment/sft_policy.json"))
    pol = Policy(sft_logits)
    ref_logits = [row[:] for row in sft_logits]

    def ref_logp(c, a):
        return math.log(softmax(ref_logits[c])[a] + 1e-30)

    data = sample_preferences(n=800, seed=seed)
    print(f"[DPO] start  E[r*] = {expected_true_reward(pol):+.3f}")

    for ep in range(epochs):
        random.shuffle(data)
        grad = zero_grid()
        for c, aw, al in data:
            ps = pol.probs(c)
            lpw = math.log(ps[aw] + 1e-30)
            lpl = math.log(ps[al] + 1e-30)
            margin = beta * (lpw - lpl - ref_logp(c, aw) + ref_logp(c, al))
            p = sigmoid(margin)
            # L = -log σ(margin). dL/dmargin = p - 1.
            coef = (p - 1.0) * beta
            gw = grad_logp(c, aw, ps)
            gl = grad_logp(c, al, ps)
            for cc in range(N_CTX):
                for aa in range(N_ACT):
                    grad[cc][aa] += coef * (gw[cc][aa] - gl[cc][aa])
        n = len(data)
        for cc in range(N_CTX):
            for aa in range(N_ACT):
                grad[cc][aa] /= n
        pol.adam_step(grad, lr=lr)

    print(f"[DPO] done   E[r*] = {expected_true_reward(pol):+.3f}")
    return pol


if __name__ == "__main__":
    pol = train_dpo()
    show_policy(pol, "DPO policy")
    with open("alignment/dpo_policy.json", "w") as f:
        json.dump(pol.logits, f)
    print("saved -> alignment/dpo_policy.json")
