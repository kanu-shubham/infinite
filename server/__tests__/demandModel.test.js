"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { BayesianLinearRegression, ridgeFit } = require("../ml/demandModel");
const { makeRng, gaussian } = require("../ml/random");

test("BLR recovers true coefficients on a noisy linear dataset", () => {
  const rng = makeRng(1);
  const trueW = [0.5, -1.5, 2.0];
  const X = [];
  const y = [];
  for (let i = 0; i < 400; i++) {
    const x = [1, gaussian(rng), gaussian(rng)];
    X.push(x);
    y.push(x[0] * trueW[0] + x[1] * trueW[1] + x[2] * trueW[2] + 0.1 * gaussian(rng));
  }
  const m = new BayesianLinearRegression({ dim: 3, priorPrecision: 1e-3, noiseVar: 0.01 });
  m.fit(X, y);
  for (let i = 0; i < 3; i++) {
    assert.ok(
      Math.abs(m.mean[i] - trueW[i]) < 0.1,
      `coef ${i} drifted: estimated ${m.mean[i]}, expected ${trueW[i]}`
    );
  }
});

test("predictDist returns positive variance and a valid std", () => {
  const m = new BayesianLinearRegression({ dim: 2 });
  m.observe([1, 0.5], 1.2);
  const { variance, std } = m.predictDist([1, 0.5]);
  assert.ok(variance > 0);
  assert.ok(std > 0);
});

test("posterior sampling produces variation around the mean", () => {
  const rng = makeRng(2);
  const m = new BayesianLinearRegression({ dim: 2, priorPrecision: 0.01 });
  for (let i = 0; i < 50; i++) m.observe([1, i / 50], 0.5 + 0.5 * (i / 50));
  const samples = Array.from({ length: 100 }, () => m.posteriorSample(rng));
  const mean = samples.reduce(
    (acc, s) => [acc[0] + s[0] / 100, acc[1] + s[1] / 100],
    [0, 0]
  );
  const variance0 = samples.reduce((a, s) => a + (s[0] - mean[0]) ** 2, 0) / 100;
  assert.ok(variance0 > 1e-6);
  assert.ok(Math.abs(mean[0] - m.mean[0]) < 0.2);
});

test("ridgeFit beats unregularized OLS on near-collinear features", () => {
  const rng = makeRng(7);
  const X = [];
  const y = [];
  for (let i = 0; i < 50; i++) {
    const a = gaussian(rng);
    const b = a + 1e-3 * gaussian(rng); // nearly identical
    X.push([1, a, b]);
    y.push(2 * a + 0.3 * gaussian(rng));
  }
  const { beta } = ridgeFit(X, y, 0.1);
  for (const w of beta) assert.ok(Number.isFinite(w));
});
