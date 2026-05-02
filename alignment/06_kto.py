"""
Stage 6 — Kahneman-Tversky Optimization (KTO).

Why: DPO/IPO need PAIRED preferences (a_w vs a_l for the same prompt).
That's expensive. KTO (Ethayarajh et al. 2024) lets you train on UNPAIRED
binary labels — "this response is desirable" / "this response is undesirable".
A real chat log naturally provides this (thumbs up / thumbs down).

Loss (one prompt+response at a time, with reference model π_ref):
    r_θ(c, a) = log( π_θ(a|c) / π_ref(a|c) )      # implicit reward
    z_ref    = E[ β * KL(π_θ || π_ref) ]          # running baseline (>=0)

    desirable :  L = w_D * ( 1 - σ( β * r_θ - z_ref ) )
    undesirable: L = w_U * ( 1 - σ( z_ref - β * r_θ ) )

Intuition: the loss is a Kahneman-Tversky-style value function — gains
above the reference baseline are rewarded, losses below it are penalized,
with diminishing sensitivity (the σ saturates).
"""
import json
import math
import random

from common import (
    N_ACT, N_CTX, TRUE_REWARD, Policy, expected_true_reward, grad_logp,
    show_policy, sigmoid, softmax, zero_grid,
)


def build_binary_data(n=1200, seed=0):
    """
    Roll out from the SFT policy and label each (c, a) as desirable iff
    the true reward exceeds the per-context median. Approximates a
    thumbs-up/down log.
    """
    rng = random.Random(seed)
    medians = [sorted(row)[len(row) // 2] for row in TRUE_REWARD]
    sft_logits = json.load(open("alignment/sft_policy.json"))
    sft = Policy(sft_logits)
    data = []
    for _ in range(n):
        c = rng.randrange(N_CTX)
        # use sft.sample but with rng for determinism via random's global state
        ps = sft.probs(c)
        u = rng.random()
        s = 0.0
        a = N_ACT - 1
        for ai, p in enumerate(ps):
            s += p
            if u < s:
                a = ai
                break
        # noisy label
        score = TRUE_REWARD[c][a] - medians[c]
        good = sigmoid(2.0 * score) > rng.random()
        data.append((c, a, 1 if good else 0))
    return data


def train_kto(beta=0.1, epochs=40, lr=0.05, w_D=1.0, w_U=1.0, seed=0):
    random.seed(seed)
    sft_logits = json.load(open("alignment/sft_policy.json"))
    pol = Policy(sft_logits)
    ref_logits = [row[:] for row in sft_logits]

    def ref_logp(c, a):
        return math.log(softmax(ref_logits[c])[a] + 1e-30)

    data = build_binary_data(seed=seed)
    print(f"[KTO] start  E[r*] = {expected_true_reward(pol):+.3f}")

    # Running estimate of z_ref = β * E[KL(π_θ || π_ref)] over recent batches.
    z_ref = 0.0
    z_decay = 0.95

    for ep in range(epochs):
        random.shuffle(data)
        # update z_ref from a small monte-carlo estimate
        kl_mc = 0.0
        for _ in range(64):
            c = random.randrange(N_CTX)
            ps_t = pol.probs(c)
            ps_r = softmax(ref_logits[c])
            kl_mc += sum(p * (math.log(p + 1e-30) - math.log(q + 1e-30))
                         for p, q in zip(ps_t, ps_r))
        kl_mc /= 64
        z_ref = z_decay * z_ref + (1 - z_decay) * beta * max(kl_mc, 0.0)

        grad = zero_grid()
        for c, a, label in data:
            ps = pol.probs(c)
            r_theta = math.log(ps[a] + 1e-30) - ref_logp(c, a)
            if label == 1:
                # L = w_D * (1 - σ(β r - z_ref))
                z = beta * r_theta - z_ref
                s = sigmoid(z)
                # dL/dr = -w_D * β * s * (1 - s)
                coef = -w_D * beta * s * (1 - s)
            else:
                # L = w_U * (1 - σ(z_ref - β r))
                z = z_ref - beta * r_theta
                s = sigmoid(z)
                # dL/dr = +w_U * β * s * (1 - s)
                coef = w_U * beta * s * (1 - s)
            glp = grad_logp(c, a, ps)
            for cc in range(N_CTX):
                for aa in range(N_ACT):
                    grad[cc][aa] += coef * glp[cc][aa]
        n = len(data)
        for cc in range(N_CTX):
            for aa in range(N_ACT):
                grad[cc][aa] /= n
        pol.adam_step(grad, lr=lr)

    print(f"[KTO] done   E[r*] = {expected_true_reward(pol):+.3f}")
    return pol


if __name__ == "__main__":
    pol = train_kto()
    show_policy(pol, "KTO policy")
    with open("alignment/kto_policy.json", "w") as f:
        json.dump(pol.logits, f)
    print("saved -> alignment/kto_policy.json")
