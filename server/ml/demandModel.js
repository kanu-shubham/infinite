"use strict";

const {
  identity,
  matAdd,
  matScale,
  matVec,
  outer,
  inverse,
  cholesky,
  dot,
  vecAdd,
  vecScale,
  transpose,
  matMul,
} = require("./math");
const { sampleMvn } = require("./random");

// Bayesian linear regression on log(units_sold + 1).
//
// Model:  y = x^T w + eps,  eps ~ N(0, noiseVar)
// Prior:  w ~ N(0, (1/priorPrecision) * I)
// Posterior closed-form:
//   precision = priorPrecision*I + (1/noiseVar) * X^T X
//   covariance = precision^{-1}
//   mean = (1/noiseVar) * covariance * X^T y
//
// We expose `posteriorSample` so the contextual bandit can do Thompson sampling
// directly off the same model, instead of fitting a separate exploration head.
class BayesianLinearRegression {
  constructor({ dim, priorPrecision = 1.0, noiseVar = 0.25 }) {
    this.dim = dim;
    this.priorPrecision = priorPrecision;
    this.noiseVar = noiseVar;
    this.precision = matScale(identity(dim), priorPrecision);
    this.precisionMean = Array(dim).fill(0); // = precision * mean
    this.mean = Array(dim).fill(0);
    this.covariance = inverse(this.precision);
    this._cholDirty = true;
    this._chol = null;
    this.n = 0;
  }

  // Online update: avoids recomputing X^T X across the whole history.
  observe(x, y) {
    if (x.length !== this.dim) throw new Error(`feature dim mismatch: expected ${this.dim}, got ${x.length}`);
    const inv = 1 / this.noiseVar;
    const xx = outer(x, x);
    this.precision = matAdd(this.precision, matScale(xx, inv));
    for (let i = 0; i < this.dim; i++) this.precisionMean[i] += inv * y * x[i];
    this.covariance = inverse(this.precision);
    this.mean = matVec(this.covariance, this.precisionMean);
    this._cholDirty = true;
    this.n += 1;
  }

  fit(X, y) {
    for (let i = 0; i < X.length; i++) this.observe(X[i], y[i]);
    return this;
  }

  predict(x) {
    return dot(this.mean, x);
  }

  // Predictive mean & variance: var = noiseVar + x^T Sigma x.
  predictDist(x) {
    const mean = this.predict(x);
    const sx = matVec(this.covariance, x);
    const variance = this.noiseVar + dot(x, sx);
    return { mean, variance, std: Math.sqrt(variance) };
  }

  posteriorSample(rng) {
    if (this._cholDirty) {
      // Add tiny jitter for numerical PD safety on degenerate posteriors.
      const jitter = 1e-9;
      const Sigma = this.covariance.map((row, i) =>
        row.map((v, j) => (i === j ? v + jitter : v))
      );
      this._chol = cholesky(Sigma);
      this._cholDirty = false;
    }
    return sampleMvn(rng, this.mean, this._chol);
  }

  toJSON() {
    return {
      dim: this.dim,
      priorPrecision: this.priorPrecision,
      noiseVar: this.noiseVar,
      mean: this.mean,
      covariance: this.covariance,
      n: this.n,
    };
  }

  static fromJSON(obj) {
    const m = new BayesianLinearRegression({
      dim: obj.dim,
      priorPrecision: obj.priorPrecision,
      noiseVar: obj.noiseVar,
    });
    m.mean = obj.mean;
    m.covariance = obj.covariance;
    m.precision = inverse(obj.covariance);
    m.precisionMean = matVec(m.precision, m.mean);
    m.n = obj.n;
    m._cholDirty = true;
    return m;
  }
}

// Closed-form ridge regression (used as a sanity baseline + warm start).
// beta = (X^T X + lambda I)^{-1} X^T y
function ridgeFit(X, y, lambda = 1e-3) {
  const n = X.length;
  const d = X[0].length;
  const Xt = transpose(X);
  const XtX = matMul(Xt, X);
  for (let i = 0; i < d; i++) XtX[i][i] += lambda;
  const Xty = matVec(Xt, y);
  const beta = matVec(inverse(XtX), Xty);
  // residual variance estimate, for noiseVar warm start
  let sse = 0;
  for (let i = 0; i < n; i++) {
    const pred = dot(X[i], beta);
    sse += (y[i] - pred) ** 2;
  }
  const sigma2 = n > d ? sse / (n - d) : sse / Math.max(n, 1);
  return { beta, sigma2 };
}

module.exports = { BayesianLinearRegression, ridgeFit };
