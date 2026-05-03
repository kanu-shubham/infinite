// Logistic-regression ranker trainer.
//
// Treats each (user, item, label) example as a pointwise sample. Features are
// the same ones the inference ranker uses, so the trained weights are
// directly usable at serve time. Loss is binary cross-entropy with L2; we
// optimise via SGD.
//
// Output artifact:
//   { weights: { [feature]: number }, bias: number, finalLoss: number }

const { mulberry32 } = require("./dataGenerator");
const { scoreMF, itemSimilarity } = require("./matrixFactorization");

const FEATURES = [
  "mfScore",         // learned: userEmb·itemEmb (CF score)
  "tagOverlap",      // content: tag overlap with user's history
  "channelHistory",  // count of previous engagements with this channel
  "popularity",      // log-popularity, normalised
  "recency",         // exp half-life freshness
];

function sigmoid(x) {
  if (x >= 0) return 1 / (1 + Math.exp(-x));
  const ex = Math.exp(x);
  return ex / (1 + ex);
}

function logloss(p, y) {
  const eps = 1e-7;
  return -(y * Math.log(p + eps) + (1 - y) * Math.log(1 - p + eps));
}

// Build a lightweight per-user history index used for feature extraction
// at training time. Mirrors what the online profile store provides at serve.
function buildHistoryIndex(events) {
  const idx = new Map(); // userId -> { tags: Map<tag,count>, channels: Map<chId,count> }
  for (const e of events) {
    if (e.label !== 1) continue; // only positives shape the user's "watched" view
    if (!idx.has(e.userId)) idx.set(e.userId, { tags: new Map(), channels: new Map(), itemIds: new Set() });
    idx.get(e.userId).itemIds.add(e.videoId);
  }
  return idx;
}

function fillHistoryFeatures(idx, catalog) {
  // After we know each user's positive items, fold their tag/channel counts.
  for (const [, hist] of idx) {
    for (const id of hist.itemIds) {
      const v = catalog.get(id);
      if (!v) continue;
      for (const t of v.tags) hist.tags.set(t, (hist.tags.get(t) || 0) + 1);
      hist.channels.set(v.channelId, (hist.channels.get(v.channelId) || 0) + 1);
    }
  }
}

function extractFeatures(event, history, catalog, mfModel) {
  const v = catalog.get(event.videoId);
  if (!v) return null;
  const hist = history.get(event.userId) || { tags: new Map(), channels: new Map(), itemIds: new Set() };

  const mfScore = mfModel ? scoreMF(mfModel, event.userId, event.videoId) : 0;

  let tagOverlap = 0;
  for (const t of v.tags) tagOverlap += hist.tags.get(t) || 0;
  tagOverlap = Math.tanh(tagOverlap / 5); // squash to [-1,1]

  const channelHistory = Math.tanh((hist.channels.get(v.channelId) || 0) / 3);

  const popularity = Math.log1p(v.popularity) / Math.log1p(60);

  const ageDays = (Date.now() - v.uploadedAt) / 86400000;
  const recency = Math.exp(-ageDays * Math.LN2 / 7);

  return { mfScore, tagOverlap, channelHistory, popularity, recency };
}

function trainLogReg(trainEvents, catalog, mfModel, opts = {}) {
  const {
    epochs = 25,
    lr = 0.1,
    reg = 0.001,
    seed = 19,
  } = opts;

  const rng = mulberry32(seed);

  // Build history index on the *training* fold only.
  const history = buildHistoryIndex(trainEvents);
  fillHistoryFeatures(history, catalog);

  // Materialise feature vectors once.
  const samples = [];
  for (const e of trainEvents) {
    const f = extractFeatures(e, history, catalog, mfModel);
    if (!f) continue;
    samples.push({ f, y: e.label });
  }
  if (samples.length === 0) throw new Error("no training samples");

  // Normalise mfScore to a comparable range so it doesn't dominate the others.
  let mu = 0, n = 0;
  for (const s of samples) { mu += s.f.mfScore; n++; }
  mu /= Math.max(1, n);
  let sigma2 = 0;
  for (const s of samples) sigma2 += (s.f.mfScore - mu) ** 2;
  const sigma = Math.sqrt(sigma2 / Math.max(1, n)) || 1;
  // Z-score and clip to [-3, 3] so a single outlier feature can't drive the
  // logit to saturation and cause train/test drift.
  for (const s of samples) {
    const z = (s.f.mfScore - mu) / sigma;
    s.f.mfScore = Math.max(-3, Math.min(3, z));
  }

  // Initialise weights at zero — logistic regression is convex, init doesn't
  // matter much but zeros are conventional.
  const weights = {};
  for (const k of FEATURES) weights[k] = 0;
  let bias = 0;

  let lastLoss = 0;
  for (let epoch = 0; epoch < epochs; epoch++) {
    // Shuffle.
    for (let i = samples.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [samples[i], samples[j]] = [samples[j], samples[i]];
    }

    let lossSum = 0;
    for (const { f, y } of samples) {
      let z = bias;
      for (const k of FEATURES) z += weights[k] * f[k];
      const p = sigmoid(z);
      lossSum += logloss(p, y);
      const err = p - y;
      for (const k of FEATURES) {
        weights[k] -= lr * (err * f[k] + reg * weights[k]);
      }
      bias -= lr * err;
    }
    lastLoss = lossSum / samples.length;
    if (epoch === 0 || (epoch + 1) % 5 === 0 || epoch === epochs - 1) {
      console.log(`  [Ranker] epoch ${epoch + 1}/${epochs}  loss=${lastLoss.toFixed(4)}`);
    }
  }

  return {
    weights,
    bias,
    mfNorm: { mu, sigma },
    finalLoss: lastLoss,
    features: FEATURES,
  };
}

function predict(rankerModel, features) {
  // Apply mfScore normalisation + clipping matching training.
  const f = { ...features };
  if (rankerModel.mfNorm) {
    const z = (f.mfScore - rankerModel.mfNorm.mu) / (rankerModel.mfNorm.sigma || 1);
    f.mfScore = Math.max(-3, Math.min(3, z));
  }
  let z = rankerModel.bias;
  for (const k of rankerModel.features) z += rankerModel.weights[k] * (f[k] || 0);
  return sigmoid(z);
}

module.exports = { trainLogReg, predict, extractFeatures, buildHistoryIndex, fillHistoryFeatures, FEATURES };
