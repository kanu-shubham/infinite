"""
Stage 8 — Knowledge Distillation (Hinton et al. 2015).

A large *teacher* model has rich knowledge in its full output distribution
(the soft probabilities), not just in the argmax label. A *student* model
trained to match the teacher's softened distribution generalizes better
than one trained on hard labels alone — the soft targets carry the
teacher's "dark knowledge" about how it ranks the *non-argmax* options.

Loss (per (c, .) example), with temperature T > 1:
    p_T = softmax(z_teacher / T)
    p_S = softmax(z_student / T)
    L_KD = T^2 * KL(p_T || p_S)

In practice we add a small CE-on-hard-labels term too. Here we use pure
KD to make the effect clean. We also force a CAPACITY GAP:
    teacher: a full N_CTX x N_ACT matrix  (any distribution).
    student: a low-rank factorization
                 logits[c, a] = U[c] · V[a],  U ∈ R^{N_CTX x r}, V ∈ R^{N_ACT x r}
             with r=2. The student is structurally limited.

You'll see KD-trained student > NLL-on-argmax student.
"""
import json
import math
import random

from common import (
    ACTIONS, CONTEXTS, N_ACT, N_CTX, Policy, expected_true_reward, show_policy,
    softmax,
)


# ---------------- Low-rank student ----------------
RANK = 2


class LowRankStudent:
    def __init__(self, rank=RANK, scale=0.3, seed=0):
        rng = random.Random(seed)
        self.U = [[rng.gauss(0, scale) for _ in range(rank)] for _ in range(N_CTX)]
        self.V = [[rng.gauss(0, scale) for _ in range(rank)] for _ in range(N_ACT)]
        self.rank = rank
        # Adam state
        self._mU = [[0.0]*rank for _ in range(N_CTX)]
        self._vU = [[0.0]*rank for _ in range(N_CTX)]
        self._mV = [[0.0]*rank for _ in range(N_ACT)]
        self._vV = [[0.0]*rank for _ in range(N_ACT)]
        self._t = 0

    def logits(self, c):
        return [sum(self.U[c][k] * self.V[a][k] for k in range(self.rank))
                for a in range(N_ACT)]

    def probs(self, c):
        return softmax(self.logits(c))

    def step(self, gU, gV, lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        for c in range(N_CTX):
            for k in range(self.rank):
                g = gU[c][k]
                self._mU[c][k] = b1*self._mU[c][k] + (1-b1)*g
                self._vU[c][k] = b2*self._vU[c][k] + (1-b2)*g*g
                m = self._mU[c][k] / (1 - b1**self._t)
                v = self._vU[c][k] / (1 - b2**self._t)
                self.U[c][k] -= lr * m / (math.sqrt(v) + eps)
        for a in range(N_ACT):
            for k in range(self.rank):
                g = gV[a][k]
                self._mV[a][k] = b1*self._mV[a][k] + (1-b1)*g
                self._vV[a][k] = b2*self._vV[a][k] + (1-b2)*g*g
                m = self._mV[a][k] / (1 - b1**self._t)
                v = self._vV[a][k] / (1 - b2**self._t)
                self.V[a][k] -= lr * m / (math.sqrt(v) + eps)


def expected_true_reward_lowrank(stu):
    from common import TRUE_REWARD
    total = 0.0
    for c in range(N_CTX):
        ps = stu.probs(c)
        total += sum(p * TRUE_REWARD[c][a] for a, p in enumerate(ps))
    return total / N_CTX


# ---------------- KD vs hard-label baselines ----------------

def grads_for_softmax_logits(loss_grad_wrt_logits, stu):
    """
    Backprop through logits[c,a] = U[c]·V[a]:
        dL/dU[c,k] = sum_a (dL/dlogits[c,a]) * V[a,k]
        dL/dV[a,k] = sum_c (dL/dlogits[c,a]) * U[c,k]
    """
    gU = [[0.0]*stu.rank for _ in range(N_CTX)]
    gV = [[0.0]*stu.rank for _ in range(N_ACT)]
    for c in range(N_CTX):
        for a in range(N_ACT):
            d = loss_grad_wrt_logits[c][a]
            for k in range(stu.rank):
                gU[c][k] += d * stu.V[a][k]
                gV[a][k] += d * stu.U[c][k]
    return gU, gV


def train_kd(teacher_logits, T=2.0, epochs=400, lr=0.05, seed=0):
    """L_KD = T^2 * KL(p_T || p_S). dL/dz_S = T * (p_S^T - p_T^T)."""
    random.seed(seed)
    stu = LowRankStudent(seed=seed)
    p_T_table = [softmax([z / T for z in teacher_logits[c]]) for c in range(N_CTX)]
    for ep in range(epochs):
        # full-batch
        loss_grad = [[0.0]*N_ACT for _ in range(N_CTX)]
        for c in range(N_CTX):
            zS = [z / T for z in stu.logits(c)]
            pS = softmax(zS)
            pT = p_T_table[c]
            # dL/dz_S[a] (where z_S = logits/T) = (pS - pT). Then chain through /T:
            # dL/dlogits[a] = (1/T) * (pS - pT). Multiply by T^2 for the loss scale:
            for a in range(N_ACT):
                loss_grad[c][a] = T * (pS[a] - pT[a])
        gU, gV = grads_for_softmax_logits(loss_grad, stu)
        stu.step(gU, gV, lr=lr)
    return stu


def train_hard(teacher_logits, epochs=400, lr=0.05, seed=0):
    """Baseline: student trained on the teacher's argmax (hard label)."""
    random.seed(seed)
    stu = LowRankStudent(seed=seed)
    targets = [max(range(N_ACT), key=lambda a: teacher_logits[c][a])
               for c in range(N_CTX)]
    for ep in range(epochs):
        loss_grad = [[0.0]*N_ACT for _ in range(N_CTX)]
        for c in range(N_CTX):
            pS = stu.probs(c)
            t = targets[c]
            # CE loss grad wrt logits = pS - onehot(t)
            for a in range(N_ACT):
                loss_grad[c][a] = pS[a] - (1.0 if a == t else 0.0)
        gU, gV = grads_for_softmax_logits(loss_grad, stu)
        stu.step(gU, gV, lr=lr)
    return stu


if __name__ == "__main__":
    teacher_logits = json.load(open("alignment/dpo_policy.json"))
    teacher = Policy(teacher_logits)
    print(f"[KD] teacher (DPO) E[r*] = {expected_true_reward(teacher):+.3f}")

    stu_hard = train_hard(teacher_logits)
    stu_kd   = train_kd(teacher_logits, T=2.0)

    print(f"[KD] student (hard labels)        E[r*] = "
          f"{expected_true_reward_lowrank(stu_hard):+.3f}")
    print(f"[KD] student (soft, T=2.0)        E[r*] = "
          f"{expected_true_reward_lowrank(stu_kd):+.3f}")

    # Show soft target effect on a specific context
    c = 1
    print(f"\n[KD] context={CONTEXTS[c]} -- teacher's soft probs at T=2:")
    pT = softmax([z / 2.0 for z in teacher_logits[c]])
    for a in range(N_ACT):
        print(f"    {ACTIONS[a]:<10} pT={pT[a]:.3f}")
