// Ranker driven by the trained logistic-regression model.
//
// Extracts the same feature set used at training time, asks the LearnedModel
// for a calibrated probability, then sorts. Output shape matches the
// heuristic ranker so the rest of the pipeline (re-ranker, slate builder)
// doesn't need to change.

const { expDecay } = require("./similarity");
const { TAG_VOCAB } = require("../data/catalog");

function buildOnlineHistory(profile, catalog) {
  // Mirror what training-time fillHistoryFeatures produced: per-user
  // tag/channel counts derived from positive engagements (= watchHistory).
  const tags = new Map();
  const channels = new Map();
  for (const id of profile.watchHistory) {
    const v = catalog.get(id);
    if (!v) continue;
    for (const t of v.tags) tags.set(t, (tags.get(t) || 0) + 1);
    channels.set(v.channelId, (channels.get(v.channelId) || 0) + 1);
  }
  return { tags, channels };
}

function extractOnlineFeatures(candidate, profile, catalog, learnedModel, hist) {
  const v = candidate.video;

  // For users absent from the trained user-embedding pool the raw 0 is not
  // "neutral" — it sits well below the training-time mean of mfScore. We
  // substitute the training mean so the normalised value lands at 0, which
  // is what an unknown user *should* contribute on this feature.
  const knownUser = learnedModel.hasUser(profile.userId);
  const mfScore = knownUser
    ? learnedModel.mfScore(profile.userId, v.id)
    : (learnedModel.ranker?.mfNorm?.mu ?? 0);

  let tagOverlap = 0;
  for (const t of v.tags) tagOverlap += hist.tags.get(t) || 0;
  tagOverlap = Math.tanh(tagOverlap / 5);

  const channelHistory = Math.tanh((hist.channels.get(v.channelId) || 0) / 3);

  const popularity = Math.log1p(v.popularity) / Math.log1p(60);

  const ageDays = (Date.now() - v.uploadedAt) / 86400000;
  const recency = expDecay(ageDays, 7);

  return { mfScore, tagOverlap, channelHistory, popularity, recency };
}

function rankWithLearned(candidates, profile, catalog, learnedModel) {
  const hist = buildOnlineHistory(profile, catalog);
  return candidates
    .map((c) => {
      const f = extractOnlineFeatures(c, profile, catalog, learnedModel, hist);
      const p = learnedModel.rankerScore(f);
      return { ...c, features: f, rankerScore: p };
    })
    .sort((a, b) => b.rankerScore - a.rankerScore);
}

function explainLearned(scored, topN = 2) {
  // Top contributing features by weight × value.
  const w = scored.features ? Object.entries(scored.features) : [];
  return w
    .map(([k, v]) => ({ feature: k, value: v }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, topN);
}

module.exports = { rankWithLearned, explainLearned, buildOnlineHistory };
