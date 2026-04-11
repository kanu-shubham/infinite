/**
 * Stage 1 — User Tower
 * ─────────────────────────────────────────────────────────────────────────────
 * Simulates a two-tower neural network's user side.
 * In production this would be a DNN that consumes raw user features (click
 * history, demographics, session context) and outputs a 128-dim embedding.
 *
 * Here we encode the user's declared preferences into the same semantic
 * space used by the item embeddings so that cosine similarity is meaningful.
 *
 * Output: Float32Array of length 128, unit-normalised.
 */

function normalize(vec) {
  const mag = Math.sqrt(vec.reduce((s, x) => s + x * x, 0)) || 1;
  return vec.map((x) => x / mag);
}

// Mirror the product embedding layout (see mockInsuranceProducts.js)
const TYPE_IDX = {
  basic: 0, comprehensive: 1, adventure: 2, medical: 3,
  cfar: 4, family: 5, senior: 6, student: 7,
};

const DEST_IDX = {
  domestic: 18, international: 19, schengen: 20, asia: 21,
  latam: 22, africa: 23, oceania: 24, adventure: 25,
  polar: 25, cruise: 24, backpacker: 23,
};

const FEAT_IDX = {
  "Trip Cancellation": 26, "Medical Emergency": 27, "Evacuation": 28,
  "Baggage Loss": 29, "Travel Delay": 30, "Adventure Sports": 31,
  "Pre-existing Conditions": 32, "Cancel For Any Reason": 33,
  "Rental Car": 34, "Equipment Loss": 35, "Search & Rescue": 36,
  "Medical Repatriation": 37, "24/7 Assistance": 38,
  "24/7 Medical Hotline": 38, "24/7 Medical Support": 38,
  "Child Medical": 39, "Pet Care Coverage": 40,
  "Electronic Equipment": 41, "High Altitude Coverage": 36,
  "Airline Failure": 33, "Trip Interruption": 26,
  "Emergency Evacuation": 28,
};

const STYLE_IDX = {
  solo: 42, couple: 43, family: 44, group: 45, business: 46,
  senior: 47, student: 48, backpacker: 49, leisure: 43, cruise: 44,
  extreme: 45, "short-trip": 42,
};

/**
 * Compute a 128-dim user embedding from a user profile.
 *
 * @param {object} userProfile  One of the mockUserProfiles entries.
 * @returns {{ embedding: number[], latencyMs: number }}
 */
export function computeUserEmbedding(userProfile) {
  const t0  = performance.now();
  const vec = new Array(128).fill(0);
  const prefs = userProfile.preferences;

  // ── [0-7] Preferred insurance types (weighted) ───────────────────────────
  for (const [i, type] of prefs.types.entries()) {
    const weight = i === 0 ? 1.0 : 0.6; // primary preference stronger
    if (TYPE_IDX[type] !== undefined) vec[TYPE_IDX[type]] = weight;
  }

  // ── [8-12] Price sensitivity (maps maxPricePerDay → tier) ───────────────
  const mpd  = prefs.maxPricePerDay;
  const tier = mpd < 3 ? 0 : mpd < 6 ? 1 : mpd < 10 ? 2 : mpd < 15 ? 3 : 4;
  // User wants products AT OR BELOW this tier → signal all lower tiers equally
  for (let t = 0; t <= tier; t++) vec[8 + t] = 1.0;

  // ── [13-17] Trip length ──────────────────────────────────────────────────
  const dur = userProfile.upcomingTrip.durationDays;
  if (dur <=  7) vec[13] = 1.0;
  if (dur <= 14) { vec[13] = 1.0; vec[14] = 1.0; }
  if (dur <= 30) { vec[14] = 1.0; vec[15] = 1.0; }
  if (dur <= 60) { vec[15] = 1.0; vec[16] = 1.0; }
  if (dur >  60) vec[16] = 1.0;
  vec[17] = 1.0;

  // ── [18-25] Destination preferences ─────────────────────────────────────
  for (const d of prefs.destinations) {
    if (DEST_IDX[d] !== undefined) vec[DEST_IDX[d]] = 1.0;
  }

  // ── [26-41] Feature preferences ─────────────────────────────────────────
  for (const [i, f] of prefs.features.entries()) {
    const weight = i < 2 ? 1.0 : 0.7; // top-2 features are most important
    if (FEAT_IDX[f] !== undefined) vec[FEAT_IDX[f]] = Math.max(vec[FEAT_IDX[f]], weight);
  }

  // ── [42-49] Travel styles ────────────────────────────────────────────────
  for (const s of prefs.styles) {
    if (STYLE_IDX[s] !== undefined) vec[STYLE_IDX[s]] = 1.0;
  }

  // ── [50-55] Implicit quality signal (users want reputable providers) ─────
  vec[50] = 0.7; // expect at least mid-tier provider
  vec[51] = 0.8; // expect decent claim satisfaction
  vec[52] = (userProfile.age > 55 ? 0.9 : 0.75); // rating sensitivity

  // ── [56-63] Coverage amount expectation ─────────────────────────────────
  const logCov = Math.log10((prefs.minCoverage || 10_000) + 1);
  const covBin = Math.min(Math.floor((logCov - 4) * 2), 7);
  for (let b = Math.max(0, covBin); b < 8; b++) vec[56 + b] = 1.0; // ≥ minCoverage

  // ── [64-127] Session / context noise (simulates DNN latent dims) ─────────
  // Seeded by user id so it's stable across pipeline runs
  let seed = Array.from(userProfile.id).reduce((a, c) => a + c.charCodeAt(0), 0);
  for (let i = 64; i < 128; i++) {
    seed = (seed * 1664525 + 1013904223) & 0xffffffff;
    vec[i] = ((seed >>> 0) / 4294967296 - 0.5) * 0.4;
  }

  const embedding = normalize(vec);
  const latencyMs = performance.now() - t0;

  return { embedding, latencyMs };
}
