"use strict";

// Decision layer: given a demand-prediction function, find the price within
// [pMin, pMax] that maximises expected business objective subject to discrete
// price grid + marginal cost. This is intentionally separate from the model
// so that policies (margin floors, max % change, regulatory caps) can be
// composed on top without touching ML internals.

function linspace(lo, hi, steps) {
  const out = new Array(steps);
  if (steps === 1) {
    out[0] = lo;
    return out;
  }
  const dx = (hi - lo) / (steps - 1);
  for (let i = 0; i < steps; i++) out[i] = lo + dx * i;
  return out;
}

function buildPriceGrid({ pMin, pMax, step = 1, snapTo = null }) {
  if (pMax < pMin) throw new Error("pMax < pMin");
  const grid = [];
  for (let p = pMin; p <= pMax + 1e-9; p += step) {
    grid.push(snapTo ? snapTo(p) : Math.round(p * 100) / 100);
  }
  return Array.from(new Set(grid));
}

// objective: 'revenue' | 'profit' | 'gmv'
// predictUnits: (price) => expected units
// marginalCost: per-unit cost for profit objective
function optimizePrice({
  prices,
  predictUnits,
  objective = "revenue",
  marginalCost = 0,
}) {
  let bestPrice = prices[0];
  let bestValue = -Infinity;
  let bestUnits = 0;
  const curve = new Array(prices.length);
  for (let i = 0; i < prices.length; i++) {
    const p = prices[i];
    const q = Math.max(0, predictUnits(p));
    let value;
    if (objective === "profit") value = (p - marginalCost) * q;
    else if (objective === "gmv") value = p * q;
    else value = p * q;
    curve[i] = { price: p, units: q, value };
    if (value > bestValue) {
      bestValue = value;
      bestPrice = p;
      bestUnits = q;
    }
  }
  return { price: bestPrice, value: bestValue, units: bestUnits, curve };
}

// Closed-form for constant-elasticity demand q = A * p^e with marginal cost c:
//   p* = c * e / (e + 1)   when e < -1
// otherwise (inelastic) revenue is unbounded, return the cap.
function closedFormElasticPrice({ elasticity, marginalCost, pMin, pMax }) {
  if (elasticity < -1) {
    const p = (marginalCost * elasticity) / (elasticity + 1);
    return Math.min(pMax, Math.max(pMin, p));
  }
  return pMax;
}

module.exports = { buildPriceGrid, optimizePrice, closedFormElasticPrice, linspace };
