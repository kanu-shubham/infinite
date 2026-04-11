/**
 * Recommendation Pipeline Orchestrator
 * ─────────────────────────────────────────────────────────────────────────────
 * Executes the four-stage pipeline and returns both the final results and
 * detailed per-stage metrics so the debug panel can visualise each step.
 *
 *   Stage 1 → User Tower      (user embedding)
 *   Stage 2 → ANN Index       (candidate retrieval)
 *   Stage 3 → Ranking Model   (re-score top 20)
 *   Stage 4 → Business Rules  (diversity, freshness, ads)
 */

import { computeUserEmbedding }  from "./userTower";
import { retrieveCandidates }    from "./annIndex";
import { rankCandidates }        from "./rankingModel";
import { applyBusinessRules }    from "./businessRules";

const RETRIEVAL_TOP_K = 500; // mirrors production "top 500 candidates"
const RANKING_TOP_N   = 20;

/**
 * @param {object}   userProfile   One of mockUserProfiles
 * @param {object[]} allProducts   Full insurance product catalogue
 * @returns {object} Pipeline result:
 *   {
 *     recommendations: object[],   // Final list shown to user
 *     stages: {
 *       userTower:     { embedding, latencyMs },
 *       annRetrieval:  { candidates, latencyMs, totalProducts },
 *       rankingModel:  { ranked, latencyMs },
 *       businessRules: { results, latencyMs, appliedRules },
 *     },
 *     totalLatencyMs: number,
 *   }
 */
export function runRecommendationPipeline(userProfile, allProducts) {
  const pipelineStart = performance.now();

  // ── Stage 1: User Tower ───────────────────────────────────────────────────
  const { embedding, latencyMs: s1Ms } = computeUserEmbedding(userProfile);

  // ── Stage 2: ANN Retrieval ────────────────────────────────────────────────
  const { candidates, latencyMs: s2Ms } = retrieveCandidates(
    embedding,
    allProducts,
    RETRIEVAL_TOP_K,
  );

  // ── Stage 3: Ranking Model ────────────────────────────────────────────────
  const { ranked, latencyMs: s3Ms } = rankCandidates(
    candidates,
    userProfile,
    RANKING_TOP_N,
  );

  // ── Stage 4: Business Rules ───────────────────────────────────────────────
  const { results, latencyMs: s4Ms, appliedRules } = applyBusinessRules(
    ranked,
    userProfile,
  );

  const totalLatencyMs = performance.now() - pipelineStart;

  return {
    recommendations: results,
    stages: {
      userTower: {
        latencyMs:   s1Ms,
        embeddingDim: embedding.length,
        description: `Encoded "${userProfile.name}" → ${embedding.length}-dim vector`,
      },
      annRetrieval: {
        latencyMs:     s2Ms,
        totalProducts: allProducts.length,
        retrieved:     candidates.length,
        description:   `Retrieved ${candidates.length} of ${allProducts.length} products (top-${RETRIEVAL_TOP_K} by cosine sim)`,
        topCandidateScore: candidates[0]?.annScore?.toFixed(4) ?? "n/a",
      },
      rankingModel: {
        latencyMs:   s3Ms,
        inputCount:  candidates.length,
        outputCount: ranked.length,
        description: `Re-scored ${candidates.length} candidates → kept top ${ranked.length}`,
        topRankScore: ranked[0]?.rankScore?.toFixed(4) ?? "n/a",
      },
      businessRules: {
        latencyMs:    s4Ms,
        finalCount:   results.length,
        appliedRules,
        description:  `Applied ${appliedRules.length} rule(s) → ${results.length} final results`,
      },
    },
    totalLatencyMs,
  };
}
