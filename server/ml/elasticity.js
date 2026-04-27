"use strict";

const { ridgeFit } = require("./demandModel");

// Constant-elasticity demand model:  log(Q) = a + e * log(P) + gamma^T z + eps
// where e is price elasticity (typically negative). For -1 < e < 0 demand is
// inelastic and revenue grows with price; for e < -1 demand is elastic and
// revenue is maximized at a finite price.
//
// Closed-form revenue-optimal price under constant elasticity (no marginal
// cost) is undefined when |e| <= 1 — the optimizer must fall back to bounds.
// With marginal cost c the lerner-rule optimum is  p* = c * e / (e + 1).
function fitConstantElasticity(samples, { lambda = 1e-3 } = {}) {
  // samples: [{ logPrice, logUnits, contextFeatures: [...] }, ...]
  if (!samples.length) {
    return { elasticity: -1.0, intercept: 0, contextWeights: [], r2: 0, n: 0 };
  }
  const ctxDim = samples[0].contextFeatures.length;
  const X = samples.map((s) => [1, s.logPrice, ...s.contextFeatures]);
  const y = samples.map((s) => s.logUnits);
  const { beta } = ridgeFit(X, y, lambda);

  let ssRes = 0;
  let ssTot = 0;
  const yMean = y.reduce((a, b) => a + b, 0) / y.length;
  for (let i = 0; i < X.length; i++) {
    let pred = 0;
    for (let j = 0; j < beta.length; j++) pred += X[i][j] * beta[j];
    ssRes += (y[i] - pred) ** 2;
    ssTot += (y[i] - yMean) ** 2;
  }
  const r2 = ssTot > 0 ? 1 - ssRes / ssTot : 0;

  return {
    intercept: beta[0],
    elasticity: beta[1],
    contextWeights: beta.slice(2),
    r2,
    n: samples.length,
    contextDim: ctxDim,
  };
}

// Predict expected units at price p given a fit and context features z.
function predictUnits(fit, price, contextFeatures) {
  const logP = Math.log(Math.max(price, 1e-6));
  let logQ = fit.intercept + fit.elasticity * logP;
  for (let i = 0; i < contextFeatures.length; i++) {
    logQ += (fit.contextWeights[i] || 0) * contextFeatures[i];
  }
  return Math.exp(logQ);
}

// Numerical (finite-difference) point-elasticity at a given price. Useful when
// the underlying demand model is not log-log but we still want to report an
// elasticity at the operating point.
function pointElasticity(predictFn, price, h = 0.01) {
  const pUp = price * (1 + h);
  const pDn = price * (1 - h);
  const qUp = predictFn(pUp);
  const qDn = predictFn(pDn);
  const qMid = predictFn(price);
  if (qMid <= 0) return 0;
  return ((qUp - qDn) / (2 * h * price)) * (price / qMid);
}

module.exports = { fitConstantElasticity, predictUnits, pointElasticity };
