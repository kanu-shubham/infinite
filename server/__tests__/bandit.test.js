"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { ThompsonPricingBandit } = require("../ml/bandit");
const { makeRng } = require("../ml/random");

test("ThompsonPricingBandit converges on the high-revenue arm", () => {
  const rng = makeRng(101);
  const bandit = new ThompsonPricingBandit({ dim: 2, priorPrecision: 0.1, noiseVar: 0.5 });

  // Two candidate prices: 100 (low units) and 200 (slightly fewer units, but
  // higher revenue). Optimal arm = 200.
  const candidates = [
    { price: 100, features: [1, Math.log(100)] },
    { price: 200, features: [1, Math.log(200)] },
  ];
  // Ground truth: log(units) = 7 + (-1.2) * log(price) — elastic, so 200 wins on revenue.
  const trueLogUnits = (price) => 7 - 1.2 * Math.log(price);

  let chose200 = 0;
  for (let i = 0; i < 600; i++) {
    const sel = bandit.selectAction(candidates, rng);
    const trueUnits = Math.max(0, Math.exp(trueLogUnits(sel.price)));
    bandit.observe(candidates[sel.index].features, trueUnits);
    if (i > 400 && sel.price === 200) chose200 += 1;
  }
  assert.ok(chose200 > 100, `Thompson failed to lock on optimum: chose200=${chose200}`);
});

test("greedy and Thompson agree once the posterior collapses", () => {
  const rng = makeRng(202);
  const bandit = new ThompsonPricingBandit({ dim: 2 });
  const candidates = [
    { price: 50, features: [1, Math.log(50)] },
    { price: 150, features: [1, Math.log(150)] },
  ];
  for (let i = 0; i < 500; i++) {
    bandit.observe(candidates[0].features, 200);
    bandit.observe(candidates[1].features, 5);
  }
  const greedy = bandit.greedyAction(candidates);
  const sampled = bandit.selectAction(candidates, rng);
  assert.equal(greedy.price, sampled.price);
});
