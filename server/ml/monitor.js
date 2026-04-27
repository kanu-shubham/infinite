"use strict";

// Population Stability Index (PSI) for distribution-shift detection.
// Rule of thumb: PSI < 0.1 stable, 0.1-0.25 moderate shift, >0.25 significant.
// Fed by feature buckets so we can monitor each input independently.
function psi(expected, actual, eps = 1e-6) {
  if (expected.length !== actual.length) {
    throw new Error("psi: bin counts must align");
  }
  const eSum = expected.reduce((a, b) => a + b, 0) || 1;
  const aSum = actual.reduce((a, b) => a + b, 0) || 1;
  let total = 0;
  for (let i = 0; i < expected.length; i++) {
    const e = expected[i] / eSum + eps;
    const a = actual[i] / aSum + eps;
    total += (a - e) * Math.log(a / e);
  }
  return total;
}

function bucketize(values, edges) {
  const counts = Array(edges.length + 1).fill(0);
  for (const v of values) {
    let placed = false;
    for (let i = 0; i < edges.length; i++) {
      if (v <= edges[i]) {
        counts[i] += 1;
        placed = true;
        break;
      }
    }
    if (!placed) counts[counts.length - 1] += 1;
  }
  return counts;
}

function classifyPsi(value) {
  if (value < 0.1) return "stable";
  if (value < 0.25) return "moderate";
  return "significant";
}

module.exports = { psi, bucketize, classifyPsi };
