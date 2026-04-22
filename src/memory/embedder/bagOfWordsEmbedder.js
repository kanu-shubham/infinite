import { cosineSimilarity, featureVector } from "../utils";

// Default embedder: sparse bag-of-words with cosine similarity. Fully local,
// zero-dependency, and serves as the baseline upgrade target for any richer
// encoder injected via `registerEmbedder` / `transformerEmbedder`.

/** @returns {import("../types").Embedder} */
export function createBagOfWordsEmbedder() {
  return {
    name: "bag-of-words",
    dim: null,
    async encode(text) {
      return featureVector(text);
    },
    sim(a, b) {
      return cosineSimilarity(a, b);
    },
    async batch(texts) {
      return texts.map((t) => featureVector(t));
    },
  };
}

export default createBagOfWordsEmbedder;
