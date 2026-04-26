// Ranker.
//
// Linear scoring model: score = Σ wᵢ · fᵢ(user, video, context). In production
// this slot is filled by a GBDT (XGBoost/LightGBM) or a deep ranker (DLRM /
// two-tower with cross). The surrounding pipeline only depends on the
// "score-each-candidate" contract, so swapping the model is local.

const { cosine, expDecay } = require("./similarity");

// Default weights, hand-set for the demo. Production weights come from
// offline training against logged labels (engagement / satisfied watches).
const DEFAULT_WEIGHTS = {
  affinity:        2.2,   // tag-vector cosine to user's tag affinity
  channelAffinity: 1.4,   // affinity for the source channel
  recency:         0.9,   // half-life-7d freshness
  popularity:      0.5,   // log-popularity, normalised
  retrievalBoost:  0.6,   // confidence from the candidate generator
};

function normalisedPopularity(v, popMax = 60) {
  return Math.log1p(v.popularity) / Math.log1p(popMax);
}

function extractFeatures(candidate, profile) {
  const v = candidate.video;
  const ageDays = (Date.now() - v.uploadedAt) / 86400000;
  return {
    affinity:        cosine(v.tagVector, profile.tagAffinity),
    channelAffinity: profile.channelAffinity[v.channelId] || 0,
    recency:         expDecay(ageDays, 7),
    popularity:      normalisedPopularity(v),
    retrievalBoost:  candidate.source === "trending" ? 0.4
                   : candidate.source === "fresh"    ? 0.3
                   : candidate.score,
  };
}

function scoreOne(candidate, profile, weights = DEFAULT_WEIGHTS) {
  const f = extractFeatures(candidate, profile);
  let score = 0;
  for (const k of Object.keys(weights)) score += weights[k] * f[k];
  return { ...candidate, features: f, rankerScore: score };
}

function rank(candidates, profile, weights = DEFAULT_WEIGHTS) {
  return candidates
    .map((c) => scoreOne(c, profile, weights))
    .sort((a, b) => b.rankerScore - a.rankerScore);
}

// Picks the top-n features by contribution. Used to power "why this video?".
function explain(scored, topN = 2) {
  return Object.entries(scored.features)
    .map(([k, v]) => ({ feature: k, value: v }))
    .sort((a, b) => b.value - a.value)
    .slice(0, topN);
}

module.exports = { rank, scoreOne, extractFeatures, explain, DEFAULT_WEIGHTS };
