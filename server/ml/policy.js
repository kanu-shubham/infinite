"use strict";

// Policy / guardrail layer. Sits between model output and the price shown to
// the user. The model proposes; this module disposes. Every adjustment is
// recorded with a reason code so audit/compliance can replay decisions.
//
// Order of operations matters:
//   1. Reject prohibited features (regulatory).
//   2. Clamp absolute floor/ceiling.
//   3. Clamp percentage change vs reference price (smoothing).
//   4. Apply margin floor (cost-plus minimum).
//   5. Apply surge cap (ethical limit during shocks).
//   6. Round to display granularity.

const PROTECTED_FEATURES = new Set([
  "race",
  "ethnicity",
  "gender",
  "religion",
  "age",
  "disability",
  "sexual_orientation",
  "national_origin",
]);

function checkProhibitedFeatures(featureNames) {
  const violations = featureNames.filter((n) => PROTECTED_FEATURES.has(n));
  if (violations.length) {
    return {
      ok: false,
      reason: `prohibited_features:${violations.join(",")}`,
    };
  }
  return { ok: true };
}

function applyGuardrails({
  proposedPrice,
  referencePrice,
  config,
  context = {},
}) {
  const adjustments = [];
  let price = proposedPrice;

  const compliance = checkProhibitedFeatures(config.featureNames || []);
  if (!compliance.ok) {
    return {
      finalPrice: referencePrice,
      adjustments: [{ reason: compliance.reason, from: proposedPrice, to: referencePrice }],
      blocked: true,
    };
  }

  if (config.absoluteFloor != null && price < config.absoluteFloor) {
    adjustments.push({ reason: "absolute_floor", from: price, to: config.absoluteFloor });
    price = config.absoluteFloor;
  }
  if (config.absoluteCeiling != null && price > config.absoluteCeiling) {
    adjustments.push({ reason: "absolute_ceiling", from: price, to: config.absoluteCeiling });
    price = config.absoluteCeiling;
  }

  if (config.maxChangePct != null && referencePrice > 0) {
    const lo = referencePrice * (1 - config.maxChangePct);
    const hi = referencePrice * (1 + config.maxChangePct);
    if (price < lo) {
      adjustments.push({ reason: "max_decrease_pct", from: price, to: lo });
      price = lo;
    } else if (price > hi) {
      adjustments.push({ reason: "max_increase_pct", from: price, to: hi });
      price = hi;
    }
  }

  if (config.marginalCost != null && config.minMarginPct != null) {
    const floor = config.marginalCost * (1 + config.minMarginPct);
    if (price < floor) {
      adjustments.push({ reason: "min_margin", from: price, to: floor });
      price = floor;
    }
  }

  if (context.surgeActive && config.surgeCapPct != null && referencePrice > 0) {
    const cap = referencePrice * (1 + config.surgeCapPct);
    if (price > cap) {
      adjustments.push({ reason: "surge_cap", from: price, to: cap });
      price = cap;
    }
  }

  const tick = config.tick || 1;
  const rounded = Math.round(price / tick) * tick;
  if (rounded !== price) {
    adjustments.push({ reason: "rounding", from: price, to: rounded });
    price = rounded;
  }

  return { finalPrice: price, adjustments, blocked: false };
}

module.exports = { applyGuardrails, checkProhibitedFeatures, PROTECTED_FEATURES };
