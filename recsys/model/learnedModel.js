// Loads trained artifacts and exposes inference primitives:
//
//   - mfItemEmbedding(itemId)            -> Float32Array | null
//   - mfUserEmbedding(userId)            -> Float32Array | null      (cold for new users)
//   - mfScore(userId, itemId)            -> number
//   - mfNearestItems(seedItemIds, n, exclude) -> [{ id, score }]
//   - rankerScore(featureBag)            -> number in [0,1]  (calibrated)
//
// All inference paths go through this module so the Recommender doesn't need
// to know about artifact format.

const fs = require("fs");
const path = require("path");
const { deserialiseMF, scoreMF, itemSimilarity } = require("../training/matrixFactorization");
const { predict: rankerPredict } = require("../training/rankerTrainer");

const ARTIFACT_DIR = path.join(__dirname, "..", "artifacts");

function loadJson(p) {
  if (!fs.existsSync(p)) return null;
  return JSON.parse(fs.readFileSync(p, "utf8"));
}

class LearnedModel {
  constructor(mfModel, rankerModel, meta) {
    this.mf = mfModel;
    this.ranker = rankerModel;
    this.meta = meta;
  }

  static loadFromDisk(dir = ARTIFACT_DIR) {
    const mfRaw = loadJson(path.join(dir, "mf.json"));
    const rankerModel = loadJson(path.join(dir, "ranker.json"));
    const meta = loadJson(path.join(dir, "meta.json"));
    if (!mfRaw || !rankerModel) return null;
    return new LearnedModel(deserialiseMF(mfRaw), rankerModel, meta);
  }

  hasUser(userId) { return this.mf && !!this.mf.userEmb[userId]; }

  mfScore(userId, itemId) {
    if (!this.mf) return 0;
    return scoreMF(this.mf, userId, itemId);
  }

  // Item-item nearest-neighbour in the learned embedding space. Used by the
  // CF retrieval generator when serving online: project the user's recent
  // watches into item space and pull NN — this works even when the online
  // user has no trained embedding.
  mfNearestItems(seedItemIds, n, excludeIds = new Set()) {
    if (!this.mf || seedItemIds.length === 0) return [];
    const scores = new Map();
    for (const seed of seedItemIds) {
      if (!this.mf.itemEmb[seed]) continue;
      for (const candId of Object.keys(this.mf.itemEmb)) {
        if (candId === seed || excludeIds.has(candId)) continue;
        const s = itemSimilarity(this.mf, seed, candId);
        if (s <= 0) continue;
        const cur = scores.get(candId);
        if (!cur || s > cur) scores.set(candId, s);
      }
    }
    const out = [];
    for (const [id, score] of scores) out.push({ id, score });
    out.sort((a, b) => b.score - a.score);
    return out.slice(0, n);
  }

  rankerScore(features) {
    return rankerPredict(this.ranker, features);
  }
}

module.exports = { LearnedModel, ARTIFACT_DIR };
