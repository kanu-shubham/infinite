"""
Shared toy world + softmax policy + manual gradients.

We use a 1-step contextual bandit instead of a real LM so every alignment
algorithm reduces to a few lines of math you can read end-to-end.

The mapping to a real LLM:
    "context"  ~ a prompt
    "action"   ~ a full response (one token in this toy)
    "policy"   ~ p(response | prompt)
    "reward"   ~ scalar quality signal from a reward model / preferences
"""
import math
import random

# -----------------------------------------------------------------------------
# Toy world
# -----------------------------------------------------------------------------

N_CTX = 4
N_ACT = 8
CONTEXTS = ["greet", "math_q", "code_q", "story_req"]
ACTIONS = ["hi", "hello!", "...", "x=2", "print(x)", "once_upon", "rude!", "off_topic"]

# True latent quality r*(c, a). Higher = better. Built so each context has
# 1-2 clearly preferred actions and a couple obvious bad ones.
TRUE_REWARD = [
    # hi    hello! ...   x=2   print  once  rude   off
    [ 1.5,  2.0, -0.5, -0.8, -0.7, -0.3, -2.0, -1.0],  # greet
    [-0.4, -0.2,  0.0,  2.2,  0.8, -0.5, -2.0, -1.5],  # math_q
    [-0.3, -0.2, -0.1,  0.7,  2.3, -0.4, -2.0, -1.2],  # code_q
    [-0.4, -0.3,  0.1, -0.3, -0.2,  2.4, -2.0, -1.0],  # story_req
]


# -----------------------------------------------------------------------------
# Math helpers
# -----------------------------------------------------------------------------

def softmax(xs):
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    s = sum(es)
    return [e / s for e in es]


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def kl(p, q):
    """KL(p || q) for categorical distributions."""
    return sum(pi * math.log((pi + 1e-30) / (qi + 1e-30)) for pi, qi in zip(p, q))


# -----------------------------------------------------------------------------
# Categorical softmax policy with analytic gradients
# -----------------------------------------------------------------------------

def zero_grid():
    return [[0.0] * N_ACT for _ in range(N_CTX)]


class Policy:
    """π(a|c) = softmax(logits[c])[a]. Trained with Adam on a manual gradient."""

    def __init__(self, init_logits=None):
        if init_logits is None:
            self.logits = zero_grid()
        else:
            self.logits = [row[:] for row in init_logits]
        self._reset_adam()

    def _reset_adam(self):
        self.m = zero_grid()
        self.v = zero_grid()
        self.t = 0

    def clone(self):
        return Policy(self.logits)

    def probs(self, c):
        return softmax(self.logits[c])

    def logp(self, c, a):
        return math.log(self.probs(c)[a] + 1e-30)

    def sample(self, c):
        ps = self.probs(c)
        r = random.random()
        s = 0.0
        for a, p in enumerate(ps):
            s += p
            if r < s:
                return a
        return N_ACT - 1

    def adam_step(self, grad, lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
        """Subtracts grad. To MAXIMIZE an objective, pass -grad_of_objective."""
        self.t += 1
        for c in range(N_CTX):
            for a in range(N_ACT):
                g = grad[c][a]
                self.m[c][a] = b1 * self.m[c][a] + (1 - b1) * g
                self.v[c][a] = b2 * self.v[c][a] + (1 - b2) * g * g
                mhat = self.m[c][a] / (1 - b1 ** self.t)
                vhat = self.v[c][a] / (1 - b2 ** self.t)
                self.logits[c][a] -= lr * mhat / (math.sqrt(vhat) + eps)


def grad_logp(c, a, probs_c):
    """
    ∇_{logits} log π(a|c).

    For softmax: ∂ log π(a|c) / ∂ logits[c'][a'] =
        1[c'==c] * (1[a'==a] - π(a'|c)).
    Returns a (N_CTX, N_ACT) grid.
    """
    g = zero_grid()
    for ap in range(N_ACT):
        g[c][ap] = (1.0 if ap == a else 0.0) - probs_c[ap]
    return g


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def expected_true_reward(pol):
    """E_{c~Unif} E_{a~π(.|c)} r*(c, a)."""
    total = 0.0
    for c in range(N_CTX):
        ps = pol.probs(c)
        total += sum(p * TRUE_REWARD[c][a] for a, p in enumerate(ps))
    return total / N_CTX


def show_policy(pol, label=""):
    print(f"-- {label} --")
    for c in range(N_CTX):
        ps = pol.probs(c)
        top = sorted(range(N_ACT), key=lambda a: -ps[a])[:3]
        parts = [f"{ACTIONS[a]}={ps[a]:.2f}" for a in top]
        print(f"  {CONTEXTS[c]:>10}: " + ", ".join(parts))


# -----------------------------------------------------------------------------
# Preference data generator (Bradley-Terry sampler from the true reward)
# -----------------------------------------------------------------------------

def sample_preferences(n=600, label_fn=None, seed=0):
    """
    Yields (context, winner_action, loser_action) triples.

    label_fn(c, a) -> scalar score used for Bradley-Terry. Defaults to TRUE_REWARD.
    """
    rng = random.Random(seed)
    if label_fn is None:
        label_fn = lambda c, a: TRUE_REWARD[c][a]
    data = []
    while len(data) < n:
        c = rng.randrange(N_CTX)
        a1, a2 = rng.sample(range(N_ACT), 2)
        p_win = sigmoid(label_fn(c, a1) - label_fn(c, a2))
        if rng.random() < p_win:
            data.append((c, a1, a2))
        else:
            data.append((c, a2, a1))
    return data
