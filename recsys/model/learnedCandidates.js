// Candidate generators that consume the trained MF model.
//
// Drop-in alternative to the heuristic content/collab generators. Used when
// the Recommender is constructed with a LearnedModel.

function learnedCollabCandidates(profile, catalog, learnedModel, n = 100) {
  const recent = profile.watchHistory.slice(-8);
  if (recent.length === 0) return [];
  const exclude = new Set([...profile.watchHistory, ...profile.skipped]);
  const nn = learnedModel.mfNearestItems(recent, n, exclude);
  return nn
    .map(({ id, score }) => {
      const v = catalog.get(id);
      return v ? { video: v, score, source: "mf-collab" } : null;
    })
    .filter(Boolean);
}

// If the user has a trained user-embedding, score every catalogue item
// directly with userEmb · itemEmb. This is the "two-tower" retrieval shape
// (in production: ANN over precomputed item embeddings keyed by user vec).
function learnedUserCandidates(profile, catalog, learnedModel, n = 100) {
  if (!learnedModel.hasUser(profile.userId)) return [];
  const exclude = new Set([...profile.watchHistory, ...profile.skipped]);
  const out = [];
  for (const v of catalog.all()) {
    if (exclude.has(v.id)) continue;
    const s = learnedModel.mfScore(profile.userId, v.id);
    out.push({ video: v, score: s, source: "mf-user" });
  }
  out.sort((a, b) => b.score - a.score);
  return out.slice(0, n);
}

module.exports = { learnedCollabCandidates, learnedUserCandidates };
