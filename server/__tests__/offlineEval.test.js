"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { ips, snips, doublyRobust } = require("../ml/offlineEval");

test("IPS recovers the on-policy value when target == logging", () => {
  const logs = Array.from({ length: 200 }, (_, i) => ({
    context: { id: i },
    action: 100,
    reward: 50,
    propensity: 1.0,
  }));
  const policy = () => 100;
  const r = ips({ logs, policy });
  assert.ok(Math.abs(r.value - 50) < 1e-6);
  assert.ok(r.ess > 0);
});

test("IPS down-weights unmatched actions", () => {
  const logs = [
    { context: { id: 1 }, action: 100, reward: 100, propensity: 0.5 },
    { context: { id: 2 }, action: 200, reward: 200, propensity: 0.5 },
  ];
  const policy = () => 100; // matches only first row
  const r = ips({ logs, policy });
  // (1/0.5)*100 / 2 = 100 average over n
  assert.ok(Math.abs(r.value - 100) < 1e-6);
});

test("SNIPS produces a stable estimate vs IPS", () => {
  const logs = Array.from({ length: 100 }, () => ({
    context: {},
    action: 10,
    reward: 5,
    propensity: 0.1,
  }));
  const policy = () => 10;
  const v = snips({ logs, policy });
  assert.ok(Math.abs(v.value - 5) < 1e-6);
});

test("doublyRobust falls back to baseline when nothing matches", () => {
  const logs = [{ context: {}, action: 1, reward: 0, propensity: 0.5 }];
  const policy = () => 999;
  const baseline = () => 7;
  const r = doublyRobust({ logs, policy, baseline });
  assert.equal(r.value, 7);
});
