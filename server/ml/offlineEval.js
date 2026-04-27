"use strict";

// Counterfactual evaluation of a new pricing policy from logged data.
//
// Logged tuple: { context, action (price), reward, propensity }
// IPS:    V_hat = (1/n) * sum_i  (1[pi(c_i)==a_i] / mu(a_i|c_i)) * r_i
// SNIPS:  weighted-average normalised version, lower variance, slight bias.
// DR (regression-augmented): subtract a learned reward estimate to reduce
// variance — implemented here with a caller-supplied baseline function.
//
// Propensities must be clipped to avoid blow-ups; we expose `clip` for that.

function ips({ logs, policy, clip = 0.01 }) {
  if (!logs.length) return { value: 0, ess: 0, n: 0 };
  let total = 0;
  let weightSum = 0;
  let weightSqSum = 0;
  for (const log of logs) {
    const chosen = policy(log.context);
    if (chosen !== log.action) continue;
    const propensity = Math.max(log.propensity, clip);
    const w = 1 / propensity;
    total += w * log.reward;
    weightSum += w;
    weightSqSum += w * w;
  }
  const value = total / logs.length;
  const ess = weightSum > 0 ? (weightSum * weightSum) / weightSqSum : 0;
  return { value, ess, n: logs.length };
}

function snips({ logs, policy, clip = 0.01 }) {
  if (!logs.length) return { value: 0, n: 0 };
  let num = 0;
  let den = 0;
  for (const log of logs) {
    const chosen = policy(log.context);
    if (chosen !== log.action) continue;
    const w = 1 / Math.max(log.propensity, clip);
    num += w * log.reward;
    den += w;
  }
  return { value: den > 0 ? num / den : 0, n: logs.length };
}

function doublyRobust({ logs, policy, baseline, clip = 0.01 }) {
  if (!logs.length) return { value: 0, n: 0 };
  let total = 0;
  for (const log of logs) {
    const chosen = policy(log.context);
    const baselineNew = baseline(log.context, chosen);
    let correction = 0;
    if (chosen === log.action) {
      const baselineLogged = baseline(log.context, log.action);
      const w = 1 / Math.max(log.propensity, clip);
      correction = w * (log.reward - baselineLogged);
    }
    total += baselineNew + correction;
  }
  return { value: total / logs.length, n: logs.length };
}

module.exports = { ips, snips, doublyRobust };
