"use strict";

const { BayesianLinearRegression } = require("./demandModel");
const { dot } = require("./math");

// Linear Thompson Sampling pricing bandit.
//
// Action space: a discrete grid of candidate prices for a given listing.
// Per round we:
//   1. Build feature vector phi(price, context) for each candidate.
//   2. Draw ONE posterior sample of demand weights (shared across candidates,
//      not per-candidate — this is what makes it Thompson rather than UCB).
//   3. Pick the price that maximises sampled expected revenue.
// Observed (price, units, revenue) feed back into the underlying BLR model.
class ThompsonPricingBandit {
  constructor({ dim, priorPrecision = 1.0, noiseVar = 0.25 }) {
    this.model = new BayesianLinearRegression({ dim, priorPrecision, noiseVar });
    this.dim = dim;
  }

  // candidates: [{ price, features }, ...]
  // Sampled weights are used to score every candidate; the sampler returns
  // both the chosen action and the full sampled scores so callers can log them.
  selectAction(candidates, rng) {
    const sampled = this.model.posteriorSample(rng);
    let bestIdx = 0;
    let bestScore = -Infinity;
    const scores = new Array(candidates.length);
    for (let i = 0; i < candidates.length; i++) {
      const logQ = dot(sampled, candidates[i].features);
      const expectedUnits = Math.exp(logQ);
      const score = candidates[i].price * expectedUnits;
      scores[i] = { price: candidates[i].price, score, expectedUnits };
      if (score > bestScore) {
        bestScore = score;
        bestIdx = i;
      }
    }
    return {
      index: bestIdx,
      price: candidates[bestIdx].price,
      score: bestScore,
      sampledWeights: sampled,
      scores,
    };
  }

  // Greedy best (no exploration) — used in a fraction of traffic for
  // exploit-only baselines and when guardrails forbid exploration.
  greedyAction(candidates) {
    let bestIdx = 0;
    let bestScore = -Infinity;
    for (let i = 0; i < candidates.length; i++) {
      const logQ = this.model.predict(candidates[i].features);
      const score = candidates[i].price * Math.exp(logQ);
      if (score > bestScore) {
        bestScore = score;
        bestIdx = i;
      }
    }
    return { index: bestIdx, price: candidates[bestIdx].price, score: bestScore };
  }

  // Reward signal here is log(units + 1); revenue is reconstructed at scoring
  // time. We log the demand response, not revenue, because revenue is
  // monotonic in price and would let the model "learn" trivially that high
  // prices = high revenue without observing the demand drop.
  observe(features, units) {
    const y = Math.log(Math.max(units, 0) + 1);
    this.model.observe(features, y);
  }

  toJSON() {
    return { dim: this.dim, model: this.model.toJSON() };
  }

  static fromJSON(obj) {
    const b = new ThompsonPricingBandit({ dim: obj.dim });
    b.model = BayesianLinearRegression.fromJSON(obj.model);
    return b;
  }
}

module.exports = { ThompsonPricingBandit };
