/**
 * Stage 3 — Ranking Model  (heavier neural net simulation)
 * ─────────────────────────────────────────────────────────────────────────────
 * The retrieval stage surfaces a broad set of relevant candidates quickly.
 * The ranking stage applies a much richer feature set to re-score those
 * candidates and select the top 20 for presentation.
 *
 * In production this would be a multi-layer DNN (often with cross-features,
 * attention, and DCN layers) running on GPU inference servers.
 *
 * Here we implement a calibrated scoring function that mirrors what such a
 * model learns to optimise:
 *   • Relevance  (ANN cosine score carries over)
 *   • Price fit  (does the price fit the user's budget?)
 *   • Coverage fit (does the coverage amount meet the user's floor?)
 *   • Feature overlap (what % of desired features are present?)
 *   • Trip length compatibility
 *   • Provider quality (claim satisfaction, rating)
 *   • Freshness (recently updated products are preferred)
 */

/**
 * Score a single candidate against the user profile.
 *
 * @param {object} candidate   Product enriched with annScore
 * @param {object} userProfile Traveler persona
 * @returns {number}           rankScore in [0, 1]
 */
function scoreCandidate(candidate, userProfile) {
  const prefs = userProfile.preferences;
  const weights = {
    annScore:      0.30,
    priceFit:      0.15,
    coverageFit:   0.15,
    featureOverlap:0.20,
    tripLength:    0.08,
    providerQuality:0.07,
    freshness:     0.05,
  };

  // ── 1. ANN score (already computed, pass it through) ─────────────────────
  const annScore = Math.max(0, candidate.annScore);

  // ── 2. Price fit: sigmoid around user's maxPricePerDay ───────────────────
  //    Score = 1 when price ≤ max, drops steeply above it
  const priceDelta = prefs.maxPricePerDay - candidate.pricePerDay;
  const priceFit   = 1 / (1 + Math.exp(-2 * priceDelta));

  // ── 3. Coverage fit: does coverage meet the minimum? ─────────────────────
  const coverageRatio = candidate.coverageAmount / (prefs.minCoverage || 10_000);
  const coverageFit   = Math.min(1, Math.log1p(coverageRatio) / Math.log1p(5));

  // ── 4. Feature overlap ────────────────────────────────────────────────────
  const wantedFeatures = new Set(prefs.features);
  const hasFeatures    = new Set(candidate.features);
  let overlap = 0;
  for (const f of wantedFeatures) if (hasFeatures.has(f)) overlap++;
  const featureOverlap = wantedFeatures.size > 0 ? overlap / wantedFeatures.size : 0.5;

  // ── 5. Trip length: product must support the trip duration ───────────────
  const tripLength = candidate.maxTripDays >= userProfile.upcomingTrip.durationDays ? 1.0 : 0.1;

  // ── 6. Provider quality: weighted average of rating and claim satisfaction
  const providerQuality = (candidate.rating / 5) * 0.5 + (candidate._provider?.claimScore ?? 0.8) * 0.5;

  // ── 7. Freshness ──────────────────────────────────────────────────────────
  const freshness = candidate.freshnessScore;

  // ── Weighted sum ─────────────────────────────────────────────────────────
  const rankScore =
    weights.annScore       * annScore       +
    weights.priceFit       * priceFit       +
    weights.coverageFit    * coverageFit    +
    weights.featureOverlap * featureOverlap +
    weights.tripLength     * tripLength     +
    weights.providerQuality * providerQuality +
    weights.freshness      * freshness;

  return Math.min(1, Math.max(0, rankScore));
}

/**
 * Re-score the ANN candidates with the full ranking model and return the top N.
 *
 * @param {object[]} candidates  Enriched products from ANN stage
 * @param {object}   userProfile Traveler persona
 * @param {number}   [topN=20]   Final candidates to surface
 * @returns {{ ranked: object[], latencyMs: number }}
 */
export function rankCandidates(candidates, userProfile, topN = 20) {
  const t0 = performance.now();

  const scored = candidates.map((c) => ({
    ...c,
    rankScore: scoreCandidate(c, userProfile),
    // Feature overlap for display
    featureMatchCount: [...new Set(userProfile.preferences.features)]
      .filter((f) => c.features.includes(f)).length,
    totalWantedFeatures: userProfile.preferences.features.length,
  }));

  scored.sort((a, b) => b.rankScore - a.rankScore);
  const ranked = scored.slice(0, Math.min(topN, scored.length));

  const latencyMs = performance.now() - t0;
  return { ranked, latencyMs };
}
