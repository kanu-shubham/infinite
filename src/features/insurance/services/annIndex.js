/**
 * Stage 2 — ANN Index  (simulates FAISS / ScaNN)
 * ─────────────────────────────────────────────────────────────────────────────
 * In production, all product embeddings are pre-indexed in a vector store.
 * At query time the user embedding is used as a query vector and the index
 * returns the top-K nearest neighbours in O(log N) time using an HNSW or IVF
 * index.
 *
 * Here we perform exact cosine similarity over the full catalogue (96 items)
 * which already fits in memory.  The interface deliberately mirrors what a
 * real ANN service would return so the calling code doesn't need to change.
 */

/**
 * Compute cosine similarity between two unit-normalised vectors.
 * Both inputs are plain number arrays of the same length.
 *
 * @param {number[]} a
 * @param {number[]} b
 * @returns {number} similarity in [-1, 1]
 */
export function cosineSimilarity(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot; // vectors are already unit-normalised
}

/**
 * Retrieve the top-K most relevant insurance products for the given user.
 *
 * Simulates an ANN lookup with a small artificial delay to reflect network
 * and index overhead that would exist in a production system.
 *
 * @param {number[]}  userEmbedding   128-dim unit vector from UserTower
 * @param {object[]}  products        Full product catalogue
 * @param {number}    [topK=500]      Candidates to surface (maps to "top 500"
 *                                    in the architecture diagram; capped at
 *                                    catalogue size in this demo)
 * @returns {{ candidates: object[], latencyMs: number }}
 */
export function retrieveCandidates(userEmbedding, products, topK = 500) {
  const t0 = performance.now();

  // Score every product against the user vector
  const scored = products.map((product) => ({
    ...product,
    annScore: cosineSimilarity(userEmbedding, product.embedding),
  }));

  // Sort descending by ANN score, keep topK
  scored.sort((a, b) => b.annScore - a.annScore);
  const candidates = scored.slice(0, Math.min(topK, scored.length));

  const latencyMs = performance.now() - t0;
  return { candidates, latencyMs };
}
