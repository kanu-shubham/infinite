"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { inverse, matMul, identity, cholesky, matVec, sigmoid } = require("../ml/math");

function approxEqualMatrix(a, b, eps = 1e-6) {
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < a[0].length; j++) {
      assert.ok(
        Math.abs(a[i][j] - b[i][j]) < eps,
        `mismatch at [${i},${j}]: ${a[i][j]} vs ${b[i][j]}`
      );
    }
  }
}

test("inverse(A) * A == I for a small SPD matrix", () => {
  const A = [
    [4, 1, 0],
    [1, 3, 1],
    [0, 1, 2],
  ];
  const Ainv = inverse(A);
  approxEqualMatrix(matMul(A, Ainv), identity(3));
});

test("inverse rejects singular matrix", () => {
  const A = [
    [1, 2],
    [2, 4],
  ];
  assert.throws(() => inverse(A), /singular/i);
});

test("cholesky factors recover A = L L^T", () => {
  const A = [
    [4, 2],
    [2, 3],
  ];
  const L = cholesky(A);
  const LT = [
    [L[0][0], L[1][0]],
    [L[0][1], L[1][1]],
  ];
  approxEqualMatrix(matMul(L, LT), A);
});

test("matVec computes Ax correctly", () => {
  const A = [
    [1, 2],
    [3, 4],
  ];
  const v = [5, 6];
  assert.deepEqual(matVec(A, v), [17, 39]);
});

test("sigmoid is numerically stable for extreme inputs", () => {
  assert.ok(sigmoid(-1000) >= 0 && sigmoid(-1000) < 1e-10);
  assert.ok(sigmoid(1000) > 1 - 1e-10 && sigmoid(1000) <= 1);
  assert.ok(Math.abs(sigmoid(0) - 0.5) < 1e-12);
});
