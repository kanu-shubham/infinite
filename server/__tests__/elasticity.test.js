"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { fitConstantElasticity, predictUnits, pointElasticity } = require("../ml/elasticity");
const { makeRng, gaussian } = require("../ml/random");

test("recovers a known elasticity from synthetic log-log data", () => {
  const rng = makeRng(11);
  const trueE = -1.7;
  const samples = [];
  for (let i = 0; i < 500; i++) {
    const price = 50 + rng() * 250;
    const ctx = [rng(), rng()];
    const logUnits = 6 + trueE * Math.log(price) + 0.05 * gaussian(rng);
    samples.push({
      logPrice: Math.log(price),
      logUnits,
      contextFeatures: ctx,
    });
  }
  const fit = fitConstantElasticity(samples);
  assert.ok(
    Math.abs(fit.elasticity - trueE) < 0.1,
    `elasticity off: ${fit.elasticity} vs ${trueE}`
  );
  assert.ok(fit.r2 > 0.9, `R^2 too low: ${fit.r2}`);
});

test("predictUnits monotonically decreases in price for negative elasticity", () => {
  const fit = { elasticity: -1.4, intercept: 5, contextWeights: [] };
  const q1 = predictUnits(fit, 100, []);
  const q2 = predictUnits(fit, 200, []);
  assert.ok(q2 < q1);
});

test("pointElasticity recovers constant elasticity numerically", () => {
  const e = -1.3;
  const predict = (p) => Math.exp(5 + e * Math.log(p));
  const numeric = pointElasticity(predict, 120);
  assert.ok(Math.abs(numeric - e) < 0.02);
});

test("fitConstantElasticity is robust to an empty input", () => {
  const fit = fitConstantElasticity([]);
  assert.equal(fit.n, 0);
  assert.ok(Number.isFinite(fit.elasticity));
});
