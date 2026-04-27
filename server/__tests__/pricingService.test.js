"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const config = require("../config");
const { PricingService } = require("../pricingService");

test("PricingService bootstrap fits demand, conversion, and elasticity", () => {
  const svc = new PricingService(config);
  svc.bootstrap();
  assert.ok(svc.repo.listHotels().length > 0);
  assert.ok(svc.repo.feedback.length > 0);
  assert.ok(svc.elasticityFits.size > 0);
  for (const fit of svc.elasticityFits.values()) {
    assert.ok(fit.elasticity < -0.5, `expected negative elasticity, got ${fit.elasticity}`);
  }
});

test("quote returns a price within absolute floor/ceiling and records audit", () => {
  const svc = new PricingService(config);
  svc.bootstrap();
  const hotel = svc.repo.listHotels()[0];
  const r = svc.quote({ hotelId: hotel.id, userId: "test-user", context: { nights: 2 } });
  assert.ok(r.finalPrice >= config.pricing.absoluteFloor);
  assert.ok(r.finalPrice <= config.pricing.absoluteCeiling);
  assert.ok(svc.repo.audit.find((a) => a.decisionId === r.decisionId));
});

test("recordFeedback updates demand model online", () => {
  const svc = new PricingService(config);
  svc.bootstrap();
  const hotel = svc.repo.listHotels()[0];
  const before = svc.demandModel.n;
  svc.recordFeedback({
    decisionId: "manual",
    hotelId: hotel.id,
    price: hotel.basePrice,
    units: 3,
    booked: true,
    context: { nights: 1 },
  });
  assert.equal(svc.demandModel.n, before + 1);
});

test("explain returns a downward-sloping demand curve overall", () => {
  const svc = new PricingService(config);
  svc.bootstrap();
  const hotel = svc.repo.listHotels()[0];
  const r = svc.explain({ hotelId: hotel.id, context: { nights: 1 } });
  const lo = r.curve[0];
  const hi = r.curve[r.curve.length - 1];
  assert.ok(hi.expectedUnits <= lo.expectedUnits + 1e-6);
  assert.ok(r.fittedElasticity < 0);
});

test("evaluatePolicy returns finite IPS / SNIPS estimates", () => {
  const svc = new PricingService(config);
  svc.bootstrap();
  const r = svc.evaluatePolicy({ uplift: 0.05 });
  assert.ok(Number.isFinite(r.ips.value));
  assert.ok(Number.isFinite(r.snips.value));
});

test("assignExperiment is deterministic per unit", () => {
  const svc = new PricingService(config);
  svc.bootstrap();
  svc.repo.setExperiment("ab1", {
    variants: [
      { name: "a", weight: 1 },
      { name: "b", weight: 1 },
    ],
  });
  const a = svc.assignExperiment("ab1", "user-42");
  const b = svc.assignExperiment("ab1", "user-42");
  assert.equal(a, b);
});
