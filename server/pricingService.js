"use strict";

const crypto = require("node:crypto");

const { Repository } = require("./store/repository");
const { seedHotels, generateHistory } = require("./store/seed");
const {
  buildContextFeatures,
  buildDemandFeatures,
  DEMAND_DIM,
  CONTEXT_DIM,
  DEMAND_FEATURE_NAMES,
} = require("./ml/features");
const { BayesianLinearRegression, ridgeFit } = require("./ml/demandModel");
const { fitConstantElasticity, predictUnits, pointElasticity } = require("./ml/elasticity");
const { LogisticRegression } = require("./ml/conversionModel");
const { ThompsonPricingBandit } = require("./ml/bandit");
const { buildPriceGrid, optimizePrice } = require("./ml/optimizer");
const { applyGuardrails } = require("./ml/policy");
const { ips, snips } = require("./ml/offlineEval");
const { psi, bucketize, classifyPsi } = require("./ml/monitor");
const { makeRng, hashBucket } = require("./ml/random");

class PricingService {
  constructor(config) {
    this.config = config;
    this.repo = new Repository();
    this.demandModel = new BayesianLinearRegression({
      dim: DEMAND_DIM,
      priorPrecision: 1.0,
      noiseVar: 0.4,
    });
    this.bandit = new ThompsonPricingBandit({
      dim: DEMAND_DIM,
      priorPrecision: config.bandit.priorPrecision,
      noiseVar: config.bandit.noiseVar,
    });
    this.conversionModel = new LogisticRegression({
      dim: CONTEXT_DIM + 1, // context + log(price)
      l2: 0.5,
      lr: 0.2,
      epochs: 250,
    });
    this.elasticityFits = new Map(); // hotelId -> fit
    this.rng = makeRng(20260427);
    this.referenceFeatureBuckets = null;
  }

  bootstrap() {
    seedHotels(this.repo, this.config.seed.hotels);
    generateHistory(this.repo, this.config.seed.samplesPerHotel);
    this.trainAll();
    this._snapshotReferenceFeatures();
  }

  // Train demand, conversion, bandit, and per-hotel elasticity from logs.
  trainAll() {
    const logs = this.repo.feedback;
    if (!logs.length) return { trained: 0 };

    const X = [];
    const yLog = [];
    const Xconv = [];
    const yConv = [];

    const byHotel = new Map();
    for (const log of logs) {
      const hotel = this.repo.getHotel(log.hotelId);
      if (!hotel) continue;
      const features = buildDemandFeatures({
        hotel,
        context: log.context,
        price: log.price,
      });
      X.push(features);
      yLog.push(Math.log(Math.max(log.units, 0) + 1));

      const ctx = buildContextFeatures({ hotel, context: log.context });
      Xconv.push([...ctx, Math.log(Math.max(log.price, 1e-6))]);
      yConv.push(log.booked ? 1 : 0);

      if (!byHotel.has(log.hotelId)) byHotel.set(log.hotelId, []);
      byHotel.get(log.hotelId).push({
        logPrice: Math.log(Math.max(log.price, 1e-6)),
        logUnits: Math.log(Math.max(log.units, 0) + 1),
        contextFeatures: ctx.slice(1), // drop bias term — ridge already adds intercept implicitly through bias=1 column
      });
    }

    // Warm-start the BLR with a ridge fit, then accept observations to build a
    // proper posterior (includes covariance for Thompson sampling).
    const { beta, sigma2 } = ridgeFit(X, yLog, 1e-2);
    this.demandModel = new BayesianLinearRegression({
      dim: DEMAND_DIM,
      priorPrecision: 1.0,
      noiseVar: Math.max(sigma2, 0.05),
    });
    this.bandit = new ThompsonPricingBandit({
      dim: DEMAND_DIM,
      priorPrecision: this.config.bandit.priorPrecision,
      noiseVar: Math.max(sigma2, 0.05),
    });
    for (let i = 0; i < X.length; i++) {
      this.demandModel.observe(X[i], yLog[i]);
      this.bandit.observe(X[i], logs[i].units);
    }
    this.demandModel.mean = beta;

    this.conversionModel.fit(Xconv, yConv);

    this.elasticityFits.clear();
    for (const [hid, samples] of byHotel.entries()) {
      this.elasticityFits.set(hid, fitConstantElasticity(samples));
    }

    return {
      trained: X.length,
      hotels: byHotel.size,
      noiseVar: this.demandModel.noiseVar,
    };
  }

  // Build a price grid for a hotel based on its base price and config span.
  _gridForHotel(hotel) {
    const span = this.config.pricing.gridSpanPct;
    const lo = Math.max(this.config.pricing.absoluteFloor, hotel.basePrice * (1 - span));
    const hi = Math.min(this.config.pricing.absoluteCeiling, hotel.basePrice * (1 + span));
    const step = (hi - lo) / (this.config.pricing.gridSteps - 1);
    return buildPriceGrid({ pMin: lo, pMax: hi, step: Math.max(1, step) });
  }

  // Main entry point: produce a price for a (hotel, user, context).
  quote({ hotelId, context = {}, userId = "anon", objective }) {
    const hotel = this.repo.getHotel(hotelId);
    if (!hotel) throw new Error(`unknown hotel ${hotelId}`);

    const obj = objective || this.config.pricing.objective;
    const grid = this._gridForHotel(hotel);

    const candidates = grid.map((price) => ({
      price,
      features: buildDemandFeatures({ hotel, context, price }),
    }));

    // Per-user deterministic exploration assignment.
    const bucket = hashBucket(`${userId}:${hotelId}`);
    const explore = bucket < this.config.bandit.explorationShare;

    const selection = explore
      ? this.bandit.selectAction(candidates, this.rng)
      : this.bandit.greedyAction(candidates);
    const propensity = explore
      ? 1 / candidates.length // approximation; true Thompson propensity is integral over posterior
      : 1.0;

    // Optimization layer using the (mean) demand model — gives a deterministic
    // recommendation alongside the bandit's possibly-explored choice.
    const optimal = optimizePrice({
      prices: grid,
      predictUnits: (p) => {
        const fx = buildDemandFeatures({ hotel, context, price: p });
        return Math.exp(this.demandModel.predict(fx));
      },
      objective: obj,
      marginalCost: hotel.marginalCost || 0,
    });

    const proposed = selection.price;

    const policy = applyGuardrails({
      proposedPrice: proposed,
      referencePrice: hotel.basePrice,
      config: {
        ...this.config.pricing,
        marginalCost: hotel.marginalCost,
      },
      context: { surgeActive: !!context.surge },
    });

    // Conversion probability at the final price (for UI/explainability).
    const ctxF = buildContextFeatures({ hotel, context });
    const convFeatures = [...ctxF, Math.log(Math.max(policy.finalPrice, 1e-6))];
    const conversionProb = this.conversionModel.predictProba(convFeatures);

    const expectedUnits = Math.exp(
      this.demandModel.predict(
        buildDemandFeatures({ hotel, context, price: policy.finalPrice })
      )
    );

    const decisionId = crypto.randomUUID();
    const audit = {
      decisionId,
      hotelId,
      userId,
      basePrice: hotel.basePrice,
      proposedPrice: proposed,
      finalPrice: policy.finalPrice,
      adjustments: policy.adjustments,
      blocked: policy.blocked,
      strategy: explore ? "thompson" : "greedy",
      objective: obj,
      optimalPrice: optimal.price,
      conversionProb,
      expectedUnits,
      propensity,
    };
    this.repo.recordAudit(audit);

    return {
      decisionId,
      hotelId,
      finalPrice: policy.finalPrice,
      proposedPrice: proposed,
      basePrice: hotel.basePrice,
      strategy: explore ? "thompson_sampling" : "greedy",
      objective: obj,
      adjustments: policy.adjustments,
      blocked: policy.blocked,
      optimalPrice: optimal.price,
      conversionProb,
      expectedUnits,
      expectedRevenue: policy.finalPrice * expectedUnits,
      propensity,
    };
  }

  recordFeedback({ decisionId, hotelId, price, units, booked, context }) {
    const audit = this.repo.audit.find((a) => a.decisionId === decisionId);
    const propensity = audit ? audit.propensity : 1;
    const entry = this.repo.recordFeedback({
      decisionId,
      hotelId,
      price,
      units,
      booked: booked ? 1 : 0,
      propensity,
      context: context || {},
    });

    const hotel = this.repo.getHotel(hotelId);
    if (hotel) {
      const features = buildDemandFeatures({ hotel, context: context || {}, price });
      this.demandModel.observe(features, Math.log(Math.max(units, 0) + 1));
      this.bandit.observe(features, units);
    }
    return entry;
  }

  // Demand curve over the grid, with mean and 1σ band, plus elasticity at
  // the operating point. Used by the UI to explain a decision.
  explain({ hotelId, context = {} }) {
    const hotel = this.repo.getHotel(hotelId);
    if (!hotel) throw new Error(`unknown hotel ${hotelId}`);
    const grid = this._gridForHotel(hotel);

    const points = grid.map((price) => {
      const fx = buildDemandFeatures({ hotel, context, price });
      const dist = this.demandModel.predictDist(fx);
      const expectedUnits = Math.exp(dist.mean);
      const lo = Math.exp(dist.mean - dist.std);
      const hi = Math.exp(dist.mean + dist.std);
      return {
        price,
        expectedUnits,
        unitsLow: lo,
        unitsHigh: hi,
        expectedRevenue: price * expectedUnits,
      };
    });

    const elasticity = pointElasticity(
      (p) =>
        Math.exp(
          this.demandModel.predict(buildDemandFeatures({ hotel, context, price: p }))
        ),
      hotel.basePrice
    );

    const fit = this.elasticityFits.get(hotelId);

    return {
      hotelId,
      hotel,
      curve: points,
      pointElasticityAtBase: elasticity,
      fittedElasticity: fit ? fit.elasticity : null,
      r2: fit ? fit.r2 : null,
      featureNames: DEMAND_FEATURE_NAMES,
    };
  }

  // Counterfactual offline evaluation of a candidate "always uplift by k%" policy.
  evaluatePolicy({ uplift = 0.1 }) {
    const policy = (ctx) => {
      const targetPrice = Math.round((ctx.basePrice || 0) * (1 + uplift));
      return targetPrice;
    };

    const logs = this.repo.feedback.map((f) => {
      const hotel = this.repo.getHotel(f.hotelId);
      return {
        context: { hotelId: f.hotelId, basePrice: hotel ? hotel.basePrice : 0 },
        action: f.price,
        reward: f.price * f.units,
        propensity: f.propensity,
      };
    });

    return {
      uplift,
      ips: ips({ logs, policy }),
      snips: snips({ logs, policy }),
    };
  }

  _snapshotReferenceFeatures() {
    const samples = this.repo.feedback.slice(-500);
    if (!samples.length) return;
    const prices = samples.map((s) => s.price);
    const edges = [50, 100, 150, 200, 300, 500];
    this.referenceFeatureBuckets = { price: bucketize(prices, edges), edges };
  }

  driftReport() {
    if (!this.referenceFeatureBuckets) return { status: "no_reference" };
    const recent = this.repo.feedback.slice(-200).map((s) => s.price);
    if (!recent.length) return { status: "no_recent" };
    const actual = bucketize(recent, this.referenceFeatureBuckets.edges);
    const value = psi(this.referenceFeatureBuckets.price, actual);
    return { feature: "price", psi: value, status: classifyPsi(value) };
  }

  // Deterministic A/B assignment by stable hash. Returns variant name.
  assignExperiment(experimentId, unitId) {
    const exp = this.repo.getExperiment(experimentId);
    if (!exp) return null;
    const r = hashBucket(`${experimentId}:${unitId}`);
    let cumulative = 0;
    const totalWeight = exp.variants.reduce((a, v) => a + v.weight, 0) || 1;
    for (const v of exp.variants) {
      cumulative += v.weight / totalWeight;
      if (r <= cumulative) return v.name;
    }
    return exp.variants[exp.variants.length - 1].name;
  }
}

module.exports = { PricingService };
