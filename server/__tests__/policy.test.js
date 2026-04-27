"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { applyGuardrails, checkProhibitedFeatures } = require("../ml/policy");

const baseCfg = {
  absoluteFloor: 30,
  absoluteCeiling: 1000,
  maxChangePct: 0.2,
  tick: 1,
  featureNames: ["bias", "rating_norm"],
};

test("guardrails clamp to absolute floor and ceiling", () => {
  const lo = applyGuardrails({
    proposedPrice: 5,
    referencePrice: 10,
    config: { ...baseCfg, maxChangePct: null },
  });
  assert.equal(lo.finalPrice, 30);
  assert.ok(lo.adjustments.some((a) => a.reason === "absolute_floor"));

  const hi = applyGuardrails({
    proposedPrice: 5000,
    referencePrice: 100,
    config: { ...baseCfg, maxChangePct: null, absoluteCeiling: 1000 },
  });
  assert.equal(hi.finalPrice, 1000);
});

test("when both floor and max-change-pct apply, the tighter one wins", () => {
  const r = applyGuardrails({
    proposedPrice: 5,
    referencePrice: 100,
    config: baseCfg, // maxChangePct = 0.2 -> band [80, 120]
  });
  assert.equal(r.finalPrice, 80);
  const reasons = r.adjustments.map((a) => a.reason);
  assert.ok(reasons.includes("absolute_floor"));
  assert.ok(reasons.includes("max_decrease_pct"));
});

test("guardrails enforce max % change vs reference", () => {
  const r = applyGuardrails({
    proposedPrice: 200,
    referencePrice: 100,
    config: baseCfg,
  });
  assert.equal(r.finalPrice, 120);
  assert.ok(r.adjustments.some((a) => a.reason === "max_increase_pct"));
});

test("guardrails block prohibited features", () => {
  const r = applyGuardrails({
    proposedPrice: 100,
    referencePrice: 100,
    config: { ...baseCfg, featureNames: ["bias", "race"] },
  });
  assert.equal(r.blocked, true);
  assert.equal(r.finalPrice, 100);
});

test("guardrails apply margin floor", () => {
  const r = applyGuardrails({
    proposedPrice: 50,
    referencePrice: 100,
    config: { ...baseCfg, marginalCost: 80, minMarginPct: 0.1 },
  });
  assert.ok(r.finalPrice >= 88);
});

test("surge cap only applies when context.surgeActive is set", () => {
  const cfg = { ...baseCfg, surgeCapPct: 0.1, maxChangePct: null };
  const noSurge = applyGuardrails({
    proposedPrice: 200,
    referencePrice: 100,
    config: cfg,
    context: { surgeActive: false },
  });
  assert.equal(noSurge.finalPrice, 200);
  const surging = applyGuardrails({
    proposedPrice: 200,
    referencePrice: 100,
    config: cfg,
    context: { surgeActive: true },
  });
  assert.equal(surging.finalPrice, 110);
  assert.ok(surging.adjustments.some((a) => a.reason === "surge_cap"));
});

test("checkProhibitedFeatures detects protected attributes", () => {
  assert.equal(checkProhibitedFeatures(["bias", "rating"]).ok, true);
  assert.equal(checkProhibitedFeatures(["bias", "gender"]).ok, false);
});
