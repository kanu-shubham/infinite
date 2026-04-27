"use strict";

const { FEATURE_NAMES } = require("./ml/features");

const config = {
  port: Number(process.env.PORT || 4000),
  pricing: {
    // Display granularity (USD, snapped to nearest tick).
    tick: 1,
    // Hard floor / ceiling regardless of model output.
    absoluteFloor: 30,
    absoluteCeiling: 1500,
    // Smoothing: cap how far we can move from yesterday's price.
    maxChangePct: 0.15,
    // Cost-plus floor: marginal cost is set per-hotel, this is the global
    // minimum margin if marginalCost is provided.
    minMarginPct: 0.1,
    // During declared "surge" windows we cap the % uplift below model output.
    surgeCapPct: 0.25,
    // Price grid resolution: fraction of base price.
    gridSpanPct: 0.5,
    gridSteps: 21,
    // Default optimization objective.
    objective: "revenue",
    // Feature names presented to the policy layer for compliance check.
    featureNames: FEATURE_NAMES,
  },
  bandit: {
    // Fraction of traffic that uses Thompson sampling exploration; the rest
    // takes the greedy (exploit-only) recommendation. Lets us A/B exploration.
    explorationShare: 0.5,
    priorPrecision: 1.0,
    noiseVar: 0.25,
  },
  seed: {
    hotels: 12,
    samplesPerHotel: 60,
  },
};

module.exports = config;
