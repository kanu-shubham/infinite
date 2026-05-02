"""
Stage 7 — Constitutional AI (CAI) and RLAIF.

Two ideas, often combined:

1. Constitutional AI (Bai et al. 2022, Anthropic):
   Replace humans in the *harmlessness* loop with an LLM critic guided
   by a written constitution (a list of natural-language principles).
   Pipeline:
     a. SFT phase ("SL-CAI"): generate response, ask the model to
        critique it under a sampled principle, ask it to revise. Train
        on the (prompt, revised) pairs.
     b. RL phase ("RLAIF"): for two candidate responses, ask the model
        which better satisfies the constitution -> AI preference label
        -> train a reward model / DPO from those AI labels.

2. RLAIF (Lee et al. 2023, Google):
   The same idea, isolated: replace human preferences with AI preferences.
   Empirically often matches RLHF on summarization/helpfulness with much
   lower data cost.

In this toy: the "constitution" is a list of penalty/bonus rules over
(c, a). The "AI judge" combines a noisy version of TRUE_REWARD with the
constitution to label preferences. Then we run DPO on those AI labels.
"""
import json
import math
import random

from common import (
    ACTIONS, CONTEXTS, N_ACT, N_CTX, TRUE_REWARD, Policy, expected_true_reward,
    grad_logp, show_policy, sigmoid, softmax, zero_grid,
)


# ------- The "constitution" -------
# Each principle adds a per-(context, action) score adjustment.
PRINCIPLES = [
    ("Be polite",      lambda c, a: -3.0 if ACTIONS[a] == "rude!" else 0.0),
    ("Stay on topic",  lambda c, a: -2.0 if ACTIONS[a] == "off_topic" else 0.0),
    ("Be specific",    lambda c, a: -0.5 if ACTIONS[a] == "..." else 0.0),
]


def constitutional_score(c, a):
    return sum(p(c, a) for _, p in PRINCIPLES)


def ai_judge(c, a, rng, noise=0.4):
    """
    Approximates an LLM critic: noisy assessment of base helpfulness +
    constitution. In real CAI the "critic" is the same LLM with a
    chain-of-thought prompt; here we simulate it.
    """
    base = TRUE_REWARD[c][a] + rng.gauss(0.0, noise)
    return base + constitutional_score(c, a)


def sl_cai_revisions(n=300, seed=0):
    """
    SL-CAI step: sample (c, a), if a violates the constitution, the
    "revised" answer is the constitution-best alternative. This is what
    the SFT phase trains on.
    """
    rng = random.Random(seed)
    sft = Policy(json.load(open("alignment/sft_policy.json")))
    data = []
    for _ in range(n):
        c = rng.randrange(N_CTX)
        a = sft.sample(c)
        if constitutional_score(c, a) < 0:
            # critique + revise: pick the highest-scoring alternative
            best = max(range(N_ACT),
                       key=lambda x: TRUE_REWARD[c][x] + constitutional_score(c, x))
            data.append((c, best))
        else:
            data.append((c, a))
    return data


def sample_ai_preferences(n=800, seed=1):
    """RLAIF labels: AI judge picks winner with BT noise."""
    rng = random.Random(seed)
    data = []
    while len(data) < n:
        c = rng.randrange(N_CTX)
        a1, a2 = rng.sample(range(N_ACT), 2)
        s1 = ai_judge(c, a1, rng)
        s2 = ai_judge(c, a2, rng)
        if rng.random() < sigmoid(s1 - s2):
            data.append((c, a1, a2))
        else:
            data.append((c, a2, a1))
    return data


def train_sl_cai(epochs=20, lr=0.05, seed=0):
    """SL-CAI: SFT on revised responses produced under the constitution."""
    random.seed(seed)
    pol = Policy(json.load(open("alignment/sft_policy.json")))
    data = sl_cai_revisions(seed=seed)
    for ep in range(epochs):
        random.shuffle(data)
        for c, a in data:
            ps = pol.probs(c)
            glp = grad_logp(c, a, ps)
            grad = [[-glp[cc][aa] for aa in range(N_ACT)] for cc in range(N_CTX)]
            pol.adam_step(grad, lr=lr)
    return pol


def train_rlaif_dpo(sl_cai_pol, beta=0.5, epochs=40, lr=0.05, seed=1):
    """RLAIF + DPO: preference loss on AI-judge labels, ref = SL-CAI policy."""
    random.seed(seed)
    pol = sl_cai_pol.clone()
    pol._reset_adam()
    ref_logits = [row[:] for row in sl_cai_pol.logits]

    def ref_logp(c, a):
        return math.log(softmax(ref_logits[c])[a] + 1e-30)

    data = sample_ai_preferences(seed=seed)
    for ep in range(epochs):
        random.shuffle(data)
        grad = zero_grid()
        for c, aw, al in data:
            ps = pol.probs(c)
            lpw = math.log(ps[aw] + 1e-30)
            lpl = math.log(ps[al] + 1e-30)
            margin = beta * (lpw - lpl - ref_logp(c, aw) + ref_logp(c, al))
            p = sigmoid(margin)
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
    return pol


def report_safety(pol, label):
    """Probability of picking a constitution-violating action."""
    bad = 0.0
    for c in range(N_CTX):
        ps = pol.probs(c)
        for a, p in enumerate(ps):
            if constitutional_score(c, a) < 0:
                bad += p
    print(f"  {label}: P(violation) over contexts = {bad:.3f}")


if __name__ == "__main__":
    print("[CAI] principles:")
    for name, _ in PRINCIPLES:
        print(f"  - {name}")

    sft = Policy(json.load(open("alignment/sft_policy.json")))
    print(f"[SL-CAI] start E[r*] = {expected_true_reward(sft):+.3f}")
    sl = train_sl_cai()
    print(f"[SL-CAI] done  E[r*] = {expected_true_reward(sl):+.3f}")

    rl = train_rlaif_dpo(sl)
    print(f"[RLAIF]  done  E[r*] = {expected_true_reward(rl):+.3f}")

    print("[CAI] constitution-violation rates:")
    report_safety(sft, "SFT     ")
    report_safety(sl,  "SL-CAI  ")
    report_safety(rl,  "RLAIF   ")
    show_policy(rl, "RLAIF policy")
    with open("alignment/rlaif_policy.json", "w") as f:
        json.dump(rl.logits, f)
    print("saved -> alignment/rlaif_policy.json")
