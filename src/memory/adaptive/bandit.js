// Epsilon-greedy multi-armed bandit over a fixed arm vocabulary. Each arm
// corresponds to a retrieval signal source; the retriever feeds back
// `reward ∈ [-1, 1]` per served result and the bandit adjusts a rolling
// average per arm. Weights are derived by softmaxing the estimates.

const DEFAULTS = {
  epsilon: 0.1,
  priorN: 3,
  priorMean: 0.5,
  minTemperature: 0.25,
};

export function createEpsilonGreedyBandit(arms, options = {}) {
  if (!Array.isArray(arms) || arms.length === 0) {
    throw new Error("createEpsilonGreedyBandit: arms required");
  }
  const opts = { ...DEFAULTS, ...options };
  const state = new Map();
  for (const arm of arms) {
    state.set(arm, { n: opts.priorN, mean: opts.priorMean });
  }

  function update(arm, reward) {
    const s = state.get(arm);
    if (!s) return;
    s.n += 1;
    s.mean += (reward - s.mean) / s.n;
  }

  function weights() {
    const out = {};
    const temperature = Math.max(opts.minTemperature, 1);
    let total = 0;
    const expd = {};
    for (const [arm, s] of state) {
      const e = Math.exp(s.mean / temperature);
      expd[arm] = e;
      total += e;
    }
    for (const arm of state.keys()) {
      out[arm] = total > 0 ? expd[arm] / total : 1 / state.size;
    }
    return out;
  }

  function pick(rng = Math.random) {
    if (rng() < opts.epsilon) {
      const keys = Array.from(state.keys());
      return keys[Math.floor(rng() * keys.length)];
    }
    let best = null;
    let bestMean = -Infinity;
    for (const [arm, s] of state) {
      if (s.mean > bestMean) {
        best = arm;
        bestMean = s.mean;
      }
    }
    return best;
  }

  function snapshot() {
    const out = {};
    for (const [arm, s] of state) out[arm] = { ...s };
    return out;
  }

  function reset(armState) {
    state.clear();
    for (const arm of arms) {
      state.set(arm, { n: opts.priorN, mean: opts.priorMean });
    }
    if (armState) {
      for (const [arm, s] of Object.entries(armState)) {
        if (state.has(arm)) state.set(arm, { ...s });
      }
    }
  }

  return { arms: arms.slice(), update, weights, pick, snapshot, reset };
}

export default createEpsilonGreedyBandit;
