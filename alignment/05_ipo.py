"""
Stage 5 — Identity Preference Optimization (IPO).

DPO's loss -log σ(β * h) drives the margin h toward +∞ when the data label
is "always w > l". On noisy or near-tied preferences, this overfits and
collapses π onto the winner — even when the *true* preference probability
is closer to 0.55 than 1.0.

IPO (Azar et al., 2023, "A General Theoretical Paradigm…") replaces the
log-sigmoid with a SQUARED loss around a finite target margin:
    h = log π_θ(a_w|c) - log π_θ(a_l|c)
       - log π_ref(a_w|c) + log π_ref(a_l|c)
    L_IPO = ( h - 1 / (2β) )^2

Properties:
    + The optimum is at a *finite* h = 1/(2β); won't push π to a delta.
    + Equivalent to optimizing for the true preference probability under
      noise; more robust on near-50/50 pairs.
    - You give up the maximum-likelihood interpretation of DPO.
"""
import json
import math
import random

from common import (
    N_ACT, N_CTX, Policy, expected_true_reward, grad_logp, sample_preferences,
    show_policy, softmax, zero_grid,
)


def train_ipo(beta=0.5, epochs=40, lr=0.05, seed=0):
    random.seed(seed)
    sft_logits = json.load(open("alignment/sft_policy.json"))
    pol = Policy(sft_logits)
    ref_logits = [row[:] for row in sft_logits]

    def ref_logp(c, a):
        return math.log(softmax(ref_logits[c])[a] + 1e-30)

    target = 1.0 / (2.0 * beta)
    data = sample_preferences(n=800, seed=seed)
    print(f"[IPO] start  E[r*] = {expected_true_reward(pol):+.3f}")

    for ep in range(epochs):
        random.shuffle(data)
        grad = zero_grid()
        for c, aw, al in data:
            ps = pol.probs(c)
            lpw = math.log(ps[aw] + 1e-30)
            lpl = math.log(ps[al] + 1e-30)
            h = lpw - lpl - ref_logp(c, aw) + ref_logp(c, al)
            # L = (h - target)^2  ->  dL/dh = 2*(h - target)
            coef = 2.0 * (h - target)
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

    print(f"[IPO] done   E[r*] = {expected_true_reward(pol):+.3f}")
    return pol


if __name__ == "__main__":
    pol = train_ipo()
    show_policy(pol, "IPO policy")
    with open("alignment/ipo_policy.json", "w") as f:
        json.dump(pol.logits, f)
    print("saved -> alignment/ipo_policy.json")
