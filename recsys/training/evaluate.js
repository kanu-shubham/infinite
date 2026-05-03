// Offline evaluation.
//
// Two perspectives:
//   - Pointwise classifier quality: AUC + log-loss on (user, item, label).
//   - Top-K retrieval quality: Recall@K and NDCG@K, using the *full pipeline*
//     (MF candidates + learned ranker) to produce a ranked list per user, and
//     the held-out positives as the ground truth.
//
// AUC is computed via the rank-based formula:
//   AUC = (sum_of_ranks_of_positives - n_pos*(n_pos+1)/2) / (n_pos * n_neg)

const { predict, extractFeatures, buildHistoryIndex, fillHistoryFeatures } = require("./rankerTrainer");
const { scoreMF } = require("./matrixFactorization");

function aucROC(samples) {
  // samples: [{ score, y }]
  const sorted = samples.slice().sort((a, b) => a.score - b.score);
  let nPos = 0, nNeg = 0;
  for (const s of samples) (s.y === 1 ? nPos++ : nNeg++);
  if (nPos === 0 || nNeg === 0) return NaN;

  let rankSumPos = 0;
  // Average ranks for ties.
  let i = 0;
  while (i < sorted.length) {
    let j = i;
    while (j + 1 < sorted.length && sorted[j + 1].score === sorted[i].score) j++;
    const avgRank = (i + j) / 2 + 1; // 1-indexed
    for (let k = i; k <= j; k++) {
      if (sorted[k].y === 1) rankSumPos += avgRank;
    }
    i = j + 1;
  }
  return (rankSumPos - nPos * (nPos + 1) / 2) / (nPos * nNeg);
}

function evaluatePointwise({ trainEvents, testEvents, catalog, mfModel, rankerModel }) {
  const history = buildHistoryIndex(trainEvents);
  fillHistoryFeatures(history, catalog);

  const scored = [];
  let lossSum = 0;
  let n = 0;
  for (const e of testEvents) {
    const f = extractFeatures(e, history, catalog, mfModel);
    if (!f) continue;
    const p = predict(rankerModel, f);
    scored.push({ score: p, y: e.label });
    const eps = 1e-7;
    lossSum += -(e.label * Math.log(p + eps) + (1 - e.label) * Math.log(1 - p + eps));
    n++;
  }
  const auc = aucROC(scored);
  const logloss = lossSum / Math.max(1, n);
  return { auc, logloss, n };
}

function evaluateRetrieval({ trainEvents, testEvents, catalog, mfModel, rankerModel, k = 10 }) {
  // Group held-out positives by user.
  const heldPositives = new Map();
  for (const e of testEvents) {
    if (e.label !== 1) continue;
    if (!heldPositives.has(e.userId)) heldPositives.set(e.userId, new Set());
    heldPositives.get(e.userId).add(e.videoId);
  }

  // Items to exclude per-user from the candidate pool: anything they engaged
  // with positively in the *train* split (we shouldn't recommend known items).
  const trainPositives = new Map();
  for (const e of trainEvents) {
    if (e.label !== 1) continue;
    if (!trainPositives.has(e.userId)) trainPositives.set(e.userId, new Set());
    trainPositives.get(e.userId).add(e.videoId);
  }

  const history = buildHistoryIndex(trainEvents);
  fillHistoryFeatures(history, catalog);

  const items = catalog.all();
  const recalls = [];
  const ndcgs = [];

  for (const [userId, positives] of heldPositives) {
    if (positives.size === 0) continue;
    const exclude = trainPositives.get(userId) || new Set();

    // Score every item with the trained ranker (using MF score as a feature).
    const scored = [];
    for (const v of items) {
      if (exclude.has(v.id)) continue;
      const fakeEvent = { userId, videoId: v.id };
      const f = extractFeatures(fakeEvent, history, catalog, mfModel);
      if (!f) continue;
      const score = predict(rankerModel, f);
      scored.push({ id: v.id, score });
    }
    scored.sort((a, b) => b.score - a.score);
    const topK = scored.slice(0, k);

    // Recall@K
    let hits = 0;
    for (const t of topK) if (positives.has(t.id)) hits++;
    recalls.push(hits / positives.size);

    // NDCG@K
    let dcg = 0;
    for (let i = 0; i < topK.length; i++) {
      if (positives.has(topK[i].id)) dcg += 1 / Math.log2(i + 2);
    }
    let idcg = 0;
    const ideal = Math.min(positives.size, k);
    for (let i = 0; i < ideal; i++) idcg += 1 / Math.log2(i + 2);
    ndcgs.push(idcg > 0 ? dcg / idcg : 0);
  }

  const mean = (xs) => xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length);
  return {
    k,
    users: recalls.length,
    recallAtK: mean(recalls),
    ndcgAtK: mean(ndcgs),
  };
}

// Random baseline — sanity check that the model is doing better than chance.
function evaluateRandomBaseline({ trainEvents, testEvents, catalog, k = 10, seed = 99 }) {
  const heldPositives = new Map();
  for (const e of testEvents) {
    if (e.label !== 1) continue;
    if (!heldPositives.has(e.userId)) heldPositives.set(e.userId, new Set());
    heldPositives.get(e.userId).add(e.videoId);
  }
  const trainPositives = new Map();
  for (const e of trainEvents) {
    if (e.label !== 1) continue;
    if (!trainPositives.has(e.userId)) trainPositives.set(e.userId, new Set());
    trainPositives.get(e.userId).add(e.videoId);
  }

  // Tiny LCG so the baseline is deterministic.
  let state = seed >>> 0;
  const rand = () => { state = (state * 1664525 + 1013904223) >>> 0; return state / 4294967296; };

  const items = catalog.all();
  const recalls = [];
  for (const [userId, positives] of heldPositives) {
    if (positives.size === 0) continue;
    const exclude = trainPositives.get(userId) || new Set();
    const pool = items.filter((v) => !exclude.has(v.id));
    // Reservoir-style: random k.
    const shuffled = pool.slice();
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    let hits = 0;
    for (let i = 0; i < Math.min(k, shuffled.length); i++) {
      if (positives.has(shuffled[i].id)) hits++;
    }
    recalls.push(hits / positives.size);
  }
  const mean = (xs) => xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length);
  return { k, recallAtK: mean(recalls) };
}

module.exports = { evaluatePointwise, evaluateRetrieval, evaluateRandomBaseline };
