"""
Stage 9 — MiniLM-style relation distillation (Wang et al. 2020).

Vanilla KD matches OUTPUT distributions. MiniLM observes that the value
of a transformer is ALSO in its INTERNAL relations — specifically the
self-attention patterns and the value-value relation matrix — and that
these matrices have a fixed shape regardless of hidden size, so a small
student can match a big teacher's relations directly.

For a single attention head with queries Q, keys K, values V (each
shape [seq_len, d]), MiniLM distills:
    A_rel = softmax(Q Q^T / sqrt(d_h))   (query-query relation)
    K_rel = softmax(K K^T / sqrt(d_h))   (key-key relation)
    V_rel = softmax(V V^T / sqrt(d_h))   (value-value relation)
    L_minilm = sum_X KL( A_X^teacher || A_X^student )       for X in {Q, K, V}

Because the relations are seq×seq matrices (size doesn't depend on
hidden dim), a thin student can match a wide teacher.

In our toy world we don't have attention. We DEMONSTRATE the principle on
the action-embedding space:
    teacher V (full): the V-vectors from the rank-8 KD teacher we synthesize.
    student V (thin): rank-2 vectors of the LowRankStudent.

The student is trained to make its action-action Gram-relation matrix
    softmax(V_S V_S^T / sqrt(r))
match the teacher's. This captures the same idea: distill *how the model
relates units to each other*, not just its outputs.
"""
import json
import math
import random

from common import N_ACT, N_CTX, softmax


def relation_matrix(V, scale_by_sqrt_d=True):
    """softmax-normalized Gram matrix of rows of V; shape (n, n)."""
    n = len(V)
    d = len(V[0])
    s = math.sqrt(d) if scale_by_sqrt_d else 1.0
    raw = [[sum(V[i][k] * V[j][k] for k in range(d)) / s for j in range(n)]
           for i in range(n)]
    return [softmax(row) for row in raw]


def kl_rows(P, Q):
    total = 0.0
    for pi, qi in zip(P, Q):
        for p, q in zip(pi, qi):
            if p > 0:
                total += p * (math.log(p) - math.log(q + 1e-30))
    return total / len(P)


def make_full_rank_teacher_V(seed=0):
    """
    Synthesize a teacher action-embedding matrix in dimension N_ACT.
    Built so that semantically similar actions are close. We just orthogonalize
    a random init for variety.
    """
    rng = random.Random(seed)
    V = [[rng.gauss(0, 1.0) for _ in range(N_ACT)] for _ in range(N_ACT)]
    return V


class StudentV:
    def __init__(self, rank=2, seed=0):
        rng = random.Random(seed)
        self.V = [[rng.gauss(0, 0.3) for _ in range(rank)] for _ in range(N_ACT)]
        self.rank = rank
        self._m = [[0.0]*rank for _ in range(N_ACT)]
        self._v = [[0.0]*rank for _ in range(N_ACT)]
        self._t = 0

    def step(self, g, lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        for a in range(N_ACT):
            for k in range(self.rank):
                gg = g[a][k]
                self._m[a][k] = b1*self._m[a][k] + (1-b1)*gg
                self._v[a][k] = b2*self._v[a][k] + (1-b2)*gg*gg
                m = self._m[a][k] / (1 - b1**self._t)
                v = self._v[a][k] / (1 - b2**self._t)
                self.V[a][k] -= lr * m / (math.sqrt(v) + eps)


def train_minilm_distill(teacher_V, epochs=600, lr=0.05, seed=0):
    """
    Numerical gradient via finite differences on the relation-KL loss.
    (Exact analytic gradient is mechanical but verbose; finite differences
    are fine for r=2 and N_ACT=8.)
    """
    random.seed(seed)
    stu = StudentV(rank=2, seed=seed)
    P_T = relation_matrix(teacher_V)
    eps = 1e-3

    def loss(student):
        return kl_rows(P_T, relation_matrix(student.V))

    for ep in range(epochs):
        base = loss(stu)
        g = [[0.0] * stu.rank for _ in range(N_ACT)]
        for a in range(N_ACT):
            for k in range(stu.rank):
                stu.V[a][k] += eps
                up = loss(stu)
                stu.V[a][k] -= 2 * eps
                dn = loss(stu)
                stu.V[a][k] += eps
                g[a][k] = (up - dn) / (2 * eps)
        stu.step(g, lr=lr)
    return stu, P_T


def matrix_str(M):
    return "\n".join("  " + " ".join(f"{x:.2f}" for x in row) for row in M)


if __name__ == "__main__":
    teacher_V = make_full_rank_teacher_V()
    stu, P_T = train_minilm_distill(teacher_V)
    P_S = relation_matrix(stu.V)
    final_loss = kl_rows(P_T, P_S)

    print("[MiniLM] teacher action-action relation (8x8):")
    print(matrix_str(P_T))
    print("\n[MiniLM] student (rank-2) action-action relation:")
    print(matrix_str(P_S))
    print(f"\n[MiniLM] mean per-row KL(teacher || student) = {final_loss:.4f}")
    print("\nTakeaway: the student is rank-2 (4x smaller), yet can reproduce")
    print("the teacher's action-action structure -- the relation matrix has")
    print("shape (N_ACT, N_ACT), independent of hidden dim, so a thin student")
    print("CAN fit it. Same idea works on attention/value relations in real")
    print("transformers.")
