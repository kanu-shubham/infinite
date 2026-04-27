"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { psi, bucketize, classifyPsi } = require("../ml/monitor");

test("psi == 0 when distributions match", () => {
  const a = [10, 20, 30, 40];
  assert.ok(psi(a, a) < 1e-9);
});

test("psi increases when distributions diverge", () => {
  const e = [50, 50, 50, 50];
  const a = [100, 50, 50, 0];
  assert.ok(psi(e, a) > 0.1);
});

test("bucketize counts values per bucket edge", () => {
  const counts = bucketize([1, 5, 10, 15, 25], [10, 20]);
  assert.deepEqual(counts, [3, 1, 1]);
});

test("classifyPsi labels match the conventional thresholds", () => {
  assert.equal(classifyPsi(0.05), "stable");
  assert.equal(classifyPsi(0.15), "moderate");
  assert.equal(classifyPsi(0.5), "significant");
});
