"use strict";

// Mulberry32: fast, tiny, decent quality seedable PRNG. Deterministic across runs.
function makeRng(seed = 42) {
  let s = seed >>> 0;
  return function rng() {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Box-Muller standard normal sample.
function gaussian(rng) {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function gaussianVec(rng, n) {
  const out = Array(n);
  for (let i = 0; i < n; i++) out[i] = gaussian(rng);
  return out;
}

// Sample from N(mu, Sigma) where chol = cholesky(Sigma).
function sampleMvn(rng, mu, chol) {
  const n = mu.length;
  const z = gaussianVec(rng, n);
  const out = Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j <= i; j++) s += chol[i][j] * z[j];
    out[i] = mu[i] + s;
  }
  return out;
}

function bernoulli(rng, p) {
  return rng() < p ? 1 : 0;
}

// Stable hash -> [0,1) for deterministic A/B bucketing across servers.
function hashBucket(key) {
  let h = 2166136261;
  for (let i = 0; i < key.length; i++) {
    h ^= key.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return ((h >>> 0) % 100000) / 100000;
}

module.exports = { makeRng, gaussian, gaussianVec, sampleMvn, bernoulli, hashBucket };
