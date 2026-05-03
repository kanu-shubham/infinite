// BPR matrix factorisation for implicit feedback.
//
// Model:    score(u, i) = userEmb[u] · itemEmb[i] + itemBias[i]
// Loss:     -log σ(score(u, pos) - score(u, neg)) + λ · ||θ||²
//
// We sample a triplet (u, i+, i-) where i+ is a video the user labeled as a
// positive engagement and i- is a uniformly-sampled item the user has not
// engaged with positively. SGD over triplets — this is Rendle et al.'s BPR.
//
// Output artifact:
//   {
//     dim: number,
//     userEmb: { [userId]: Float32Array },
//     itemEmb: { [videoId]: Float32Array },
//     itemBias: { [videoId]: number },
//   }

const { mulberry32 } = require("./dataGenerator");

function sigmoid(x) {
  if (x >= 0) return 1 / (1 + Math.exp(-x));
  const ex = Math.exp(x);
  return ex / (1 + ex);
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function initVec(dim, scale, rng) {
  const v = new Float32Array(dim);
  for (let i = 0; i < dim; i++) v[i] = (rng() - 0.5) * scale;
  return v;
}

function buildPositiveIndex(events) {
  // userId -> Set of itemIds the user gave label=1 to.
  const idx = new Map();
  for (const e of events) {
    if (e.label !== 1) continue;
    if (!idx.has(e.userId)) idx.set(e.userId, new Set());
    idx.get(e.userId).add(e.videoId);
  }
  return idx;
}

function trainBPR(events, catalog, opts = {}) {
  const {
    dim = 16,
    epochs = 30,
    lr = 0.05,
    reg = 0.02,
    seed = 13,
  } = opts;

  const rng = mulberry32(seed);
  const positives = buildPositiveIndex(events);
  const userIds = [...positives.keys()];
  const itemIds = catalog.all().map((v) => v.id);
  if (userIds.length === 0) throw new Error("no positive interactions to train on");

  const userEmb = {};
  const itemEmb = {};
  const itemBias = {};
  for (const u of userIds) userEmb[u] = initVec(dim, 0.1, rng);
  for (const i of itemIds) {
    itemEmb[i] = initVec(dim, 0.1, rng);
    itemBias[i] = 0;
  }

  // Pre-flatten positive triples for efficient sampling.
  const flatPos = [];
  for (const [u, set] of positives) {
    for (const i of set) flatPos.push([u, i]);
  }

  const stepsPerEpoch = Math.max(1000, flatPos.length * 4);
  let lastLoss = 0;

  for (let epoch = 0; epoch < epochs; epoch++) {
    let lossSum = 0;
    for (let s = 0; s < stepsPerEpoch; s++) {
      // Sample (u, pos, neg).
      const [u, pos] = flatPos[Math.floor(rng() * flatPos.length)];
      const userPos = positives.get(u);
      let neg;
      // Rejection-sample a true negative (not in userPos).
      for (let tries = 0; tries < 5; tries++) {
        const cand = itemIds[Math.floor(rng() * itemIds.length)];
        if (!userPos.has(cand)) { neg = cand; break; }
      }
      if (!neg) continue;

      const uVec = userEmb[u];
      const pVec = itemEmb[pos];
      const nVec = itemEmb[neg];
      const pBias = itemBias[pos];
      const nBias = itemBias[neg];

      const xUij = (dot(uVec, pVec) + pBias) - (dot(uVec, nVec) + nBias);
      const sig = sigmoid(-xUij);  // gradient multiplier
      lossSum += Math.log(1 + Math.exp(-xUij));

      // Gradients: d/dθ [-log σ(xUij)] = -σ(-xUij) · d xUij/dθ
      for (let k = 0; k < dim; k++) {
        const gU = sig * (pVec[k] - nVec[k]) - reg * uVec[k];
        const gP = sig * uVec[k] - reg * pVec[k];
        const gN = -sig * uVec[k] - reg * nVec[k];
        uVec[k] += lr * gU;
        pVec[k] += lr * gP;
        nVec[k] += lr * gN;
      }
      itemBias[pos] = pBias + lr * (sig - reg * pBias);
      itemBias[neg] = nBias + lr * (-sig - reg * nBias);
    }
    lastLoss = lossSum / stepsPerEpoch;
    if (epoch === 0 || (epoch + 1) % 5 === 0 || epoch === epochs - 1) {
      console.log(`  [MF] epoch ${epoch + 1}/${epochs}  loss=${lastLoss.toFixed(4)}`);
    }
  }

  return { dim, userEmb, itemEmb, itemBias, finalLoss: lastLoss };
}

// Score a (user, item) pair using the trained model.
function scoreMF(model, userId, itemId) {
  const u = model.userEmb[userId];
  const i = model.itemEmb[itemId];
  if (!u || !i) return 0;
  return dot(u, i) + (model.itemBias[itemId] || 0);
}

// Item-item similarity in the learned space.
function itemSimilarity(model, idA, idB) {
  const a = model.itemEmb[idA];
  const b = model.itemEmb[idB];
  if (!a || !b) return 0;
  let dotAB = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dotAB += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i];
  }
  if (na === 0 || nb === 0) return 0;
  return dotAB / (Math.sqrt(na) * Math.sqrt(nb));
}

// Serialise to plain JSON-able object (Float32Array → Array<number>).
function serialiseMF(model) {
  const out = { dim: model.dim, userEmb: {}, itemEmb: {}, itemBias: model.itemBias, finalLoss: model.finalLoss };
  for (const u of Object.keys(model.userEmb)) out.userEmb[u] = Array.from(model.userEmb[u]);
  for (const i of Object.keys(model.itemEmb)) out.itemEmb[i] = Array.from(model.itemEmb[i]);
  return out;
}

function deserialiseMF(json) {
  const out = { dim: json.dim, userEmb: {}, itemEmb: {}, itemBias: json.itemBias || {}, finalLoss: json.finalLoss };
  for (const u of Object.keys(json.userEmb)) out.userEmb[u] = Float32Array.from(json.userEmb[u]);
  for (const i of Object.keys(json.itemEmb)) out.itemEmb[i] = Float32Array.from(json.itemEmb[i]);
  return out;
}

module.exports = { trainBPR, scoreMF, itemSimilarity, serialiseMF, deserialiseMF };
