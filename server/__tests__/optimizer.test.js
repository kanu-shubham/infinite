"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const {
  buildPriceGrid,
  optimizePrice,
  closedFormElasticPrice,
} = require("../ml/optimizer");

test("buildPriceGrid returns sorted unique values across the range", () => {
  const grid = buildPriceGrid({ pMin: 50, pMax: 100, step: 10 });
  assert.deepEqual(grid, [50, 60, 70, 80, 90, 100]);
});

test("optimizePrice picks revenue-optimal point on a unimodal demand curve", () => {
  const grid = buildPriceGrid({ pMin: 50, pMax: 200, step: 1 });
  const predictUnits = (p) => Math.exp(8 - 1.5 * Math.log(p));
  // For constant elasticity -1.5 with no marginal cost, revenue diverges as
  // price decreases — so the optimum should sit at the boundary.
  const r = optimizePrice({ prices: grid, predictUnits });
  assert.equal(r.price, 50);
});

test("optimizePrice picks the analytical optimum with marginal cost", () => {
  const grid = buildPriceGrid({ pMin: 30, pMax: 500, step: 1 });
  const e = -2.0;
  const predictUnits = (p) => Math.exp(8 + e * Math.log(p));
  const r = optimizePrice({
    prices: grid,
    predictUnits,
    objective: "profit",
    marginalCost: 50,
  });
  // Lerner rule: p* = c * e / (e+1) = 50 * -2 / -1 = 100.
  assert.ok(Math.abs(r.price - 100) <= 1);
});

test("closedFormElasticPrice respects bounds when |e| <= 1", () => {
  const cap = closedFormElasticPrice({
    elasticity: -0.5,
    marginalCost: 50,
    pMin: 30,
    pMax: 200,
  });
  assert.equal(cap, 200);
});
