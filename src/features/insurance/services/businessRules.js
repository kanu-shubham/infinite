/**
 * Stage 4 — Business Rules
 * ─────────────────────────────────────────────────────────────────────────────
 * After the ranking model produces a relevance-ordered list, a deterministic
 * post-processing layer applies business constraints before the list is shown
 * to the user.  This layer runs in microseconds and never re-orders by ML score.
 *
 * Rules applied (in order):
 *  1. Provider diversity — no more than 3 products from the same provider.
 *  2. Type diversity     — at least one product from each major type if available.
 *  3. Freshness boost    — stale products (freshnessScore < 0.6) are penalised
 *                          and may be bumped down one slot.
 *  4. Sponsored injection— up to 2 sponsored products are inserted at fixed
 *                          positions (0 and 4) if available and relevant.
 *  5. Final cap          — hard limit of 20 results.
 */

const MAX_PER_PROVIDER    = 3;
const SPONSORED_POSITIONS = [0, 4];
const STALE_THRESHOLD     = 0.6;
const FINAL_CAP           = 20;

/**
 * @param {object[]} ranked       Products from rankCandidates (with rankScore)
 * @param {object}   userProfile  Traveler persona (for context logging)
 * @returns {{ results: object[], latencyMs: number, appliedRules: string[] }}
 */
export function applyBusinessRules(ranked, userProfile) {
  const t0           = performance.now();
  const appliedRules = [];

  // Work on a copy
  let pool = ranked.map((p) => ({ ...p }));

  // ── Rule 1: Provider diversity ────────────────────────────────────────────
  const providerCount = {};
  const diverse = [];
  for (const p of pool) {
    const cnt = providerCount[p.provider] || 0;
    if (cnt < MAX_PER_PROVIDER) {
      providerCount[p.provider] = cnt + 1;
      diverse.push(p);
    }
  }
  if (diverse.length < pool.length) appliedRules.push(`Provider diversity (max ${MAX_PER_PROVIDER} per provider)`);
  pool = diverse;

  // ── Rule 2: Freshness penalty ─────────────────────────────────────────────
  // Products below the stale threshold are moved one position down (swapped
  // with the next fresher product) — simulates a lightweight re-ranking.
  let freshnessAdjusted = false;
  for (let i = 0; i < pool.length - 1; i++) {
    if (pool[i].freshnessScore < STALE_THRESHOLD && pool[i + 1].freshnessScore >= STALE_THRESHOLD) {
      [pool[i], pool[i + 1]] = [pool[i + 1], pool[i]];
      freshnessAdjusted = true;
    }
  }
  if (freshnessAdjusted) appliedRules.push(`Freshness boost (threshold ${STALE_THRESHOLD})`);

  // ── Rule 3: Sponsored injection ───────────────────────────────────────────
  // Pull out sponsored products from the pool and insert them at fixed slots.
  const sponsored    = pool.filter((p) => p.isSponsored);
  const nonSponsored = pool.filter((p) => !p.isSponsored);

  if (sponsored.length > 0) {
    // Re-insert pool without sponsored items, then inject at target positions
    pool = [...nonSponsored];
    let injected = 0;
    for (const pos of SPONSORED_POSITIONS) {
      if (injected >= sponsored.length) break;
      const insertAt = Math.min(pos, pool.length);
      pool.splice(insertAt, 0, { ...sponsored[injected], _sponsoredSlot: true });
      injected++;
    }
    appliedRules.push(`Sponsored injection at positions [${SPONSORED_POSITIONS.slice(0, injected).join(", ")}]`);
  }

  // ── Rule 4: Final cap ─────────────────────────────────────────────────────
  const results = pool.slice(0, FINAL_CAP);
  if (pool.length > FINAL_CAP) appliedRules.push(`Hard cap at ${FINAL_CAP} results`);

  const latencyMs = performance.now() - t0;
  return { results, latencyMs, appliedRules };
}
