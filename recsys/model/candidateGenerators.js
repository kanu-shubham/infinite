// Candidate generation (the "retrieval" stage).
//
// Each generator reduces the millions-of-videos catalogue to ~tens-to-hundreds
// of plausible candidates. We use four sources:
//
//   - content:  cosine(user-tag-affinity, video-tag-vector)
//   - collab:   item-item nearest-neighbour from recent watches
//   - trending: globally popular (cold-start + exploration safety net)
//   - fresh:    recent uploads (freshness + new-content discovery)
//
// Each candidate is annotated with `source` and a `score` (the source's own
// confidence) for use by the ranker.

const { cosine, expDecay } = require("./similarity");

function contentCandidates(profile, catalog, n = 100) {
  const sumAffinity = profile.tagAffinity.reduce((a, b) => a + b, 0);
  if (sumAffinity === 0) return [];
  const out = [];
  for (const v of catalog.all()) {
    const score = cosine(v.tagVector, profile.tagAffinity);
    if (score > 0) out.push({ video: v, score, source: "content" });
  }
  out.sort((a, b) => b.score - a.score);
  return out.slice(0, n);
}

// Item-item collaborative filtering, approximated by nearest-neighbour on
// content embeddings of recently-watched seeds. In production this would be
// a proper user/item embedding model (matrix factorisation or two-tower)
// served from an ANN index.
function collaborativeCandidates(profile, catalog, n = 100) {
  const recentIds = profile.watchHistory.slice(-8);
  if (recentIds.length === 0) return [];
  const seeds = recentIds.map((id) => catalog.get(id)).filter(Boolean);
  if (seeds.length === 0) return [];

  const scores = new Map();
  for (const seed of seeds) {
    for (const v of catalog.all()) {
      if (v.id === seed.id) continue;
      const s = cosine(v.tagVector, seed.tagVector);
      if (s <= 0) continue;
      const cur = scores.get(v.id);
      if (!cur || s > cur) scores.set(v.id, s);
    }
  }
  const out = [];
  for (const [id, score] of scores) {
    out.push({ video: catalog.get(id), score, source: "collab" });
  }
  out.sort((a, b) => b.score - a.score);
  return out.slice(0, n);
}

function trendingCandidates(catalog, n = 40) {
  return [...catalog.all()]
    .sort((a, b) => b.popularity - a.popularity)
    .slice(0, n)
    .map((v) => ({ video: v, score: v.popularity, source: "trending" }));
}

function freshCandidates(catalog, n = 30) {
  const now = Date.now();
  return [...catalog.all()]
    .sort((a, b) => b.uploadedAt - a.uploadedAt)
    .slice(0, n)
    .map((v) => {
      const ageDays = (now - v.uploadedAt) / 86400000;
      return { video: v, score: expDecay(ageDays, 7), source: "fresh" };
    });
}

// Union the sources, deduping by video id and keeping the best per-source
// score. Returns an array of { video, score, source }.
function generateCandidates(profile, catalog, opts = {}) {
  const {
    contentN = 100,
    collabN = 100,
    trendingN = 40,
    freshN = 30,
  } = opts;

  const pool = new Map();
  const merge = (list) => {
    for (const c of list) {
      const cur = pool.get(c.video.id);
      if (!cur || c.score > cur.score) pool.set(c.video.id, c);
    }
  };

  merge(contentCandidates(profile, catalog, contentN));
  merge(collaborativeCandidates(profile, catalog, collabN));
  merge(trendingCandidates(catalog, trendingN));
  merge(freshCandidates(catalog, freshN));

  return [...pool.values()];
}

module.exports = {
  contentCandidates,
  collaborativeCandidates,
  trendingCandidates,
  freshCandidates,
  generateCandidates,
};
