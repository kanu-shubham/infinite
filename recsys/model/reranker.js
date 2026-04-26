// Re-ranker.
//
// Greedy MMR-style diversity: when picking the next slot, penalise candidates
// from channels/tags we've already selected in the slate. This keeps the feed
// from collapsing into a single topic even when the ranker strongly prefers
// it.
//
// We also apply hard filters here: drop already-watched, already-skipped, and
// blocklisted videos. Hard filters belong upstream of the ranker conceptually
// but are cheap to apply here too as a safety net.

const DEFAULT_OPTS = {
  lambda:           0.75,  // 1 = pure relevance, 0 = pure diversity
  channelPenalty:   0.35,
  tagPenalty:       0.15,
  freshnessBoost:   0.05,  // tiny multiplicative bonus to <2-day-old uploads
};

function isFresh(video) {
  return (Date.now() - video.uploadedAt) < 2 * 86400000;
}

function diversify(rankedSlate, k, opts = {}) {
  const { lambda, channelPenalty, tagPenalty, freshnessBoost } = { ...DEFAULT_OPTS, ...opts };
  const picked = [];
  const remaining = rankedSlate.slice();
  const channelCount = new Map();
  const tagCount = new Map();

  while (picked.length < k && remaining.length > 0) {
    let bestIdx = 0;
    let bestScore = -Infinity;

    for (let i = 0; i < remaining.length; i++) {
      const c = remaining[i];
      const v = c.video;
      const chPenalty = channelCount.get(v.channelId) || 0;
      const tagPen = v.tags.reduce(
        (acc, t) => acc + (tagCount.get(t) || 0),
        0,
      );
      const penalty = channelPenalty * chPenalty + tagPenalty * tagPen;
      const freshBonus = isFresh(v) ? freshnessBoost : 0;
      const adjusted = lambda * c.rankerScore + freshBonus - (1 - lambda) * penalty;
      if (adjusted > bestScore) {
        bestScore = adjusted;
        bestIdx = i;
      }
    }

    const chosen = remaining.splice(bestIdx, 1)[0];
    chosen.finalScore = bestScore;
    picked.push(chosen);
    channelCount.set(chosen.video.channelId, (channelCount.get(chosen.video.channelId) || 0) + 1);
    chosen.video.tags.forEach((t) => tagCount.set(t, (tagCount.get(t) || 0) + 1));
  }
  return picked;
}

function applyHardFilters(candidates, { excludeIds = new Set(), blocklist = new Set() } = {}) {
  return candidates.filter(
    (c) => !excludeIds.has(c.video.id) && !blocklist.has(c.video.id),
  );
}

module.exports = { diversify, applyHardFilters, DEFAULT_OPTS };
