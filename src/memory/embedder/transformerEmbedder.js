import { cosineSimilarity } from "../utils";

// Adapter for user-provided dense encoders. The consumer injects an async
// function that turns text into a Float32Array (e.g. transformers.js, ONNX,
// a remote embedding service). This module handles caching, batching, and
// cosine similarity for dense vectors.

/**
 * @param {{
 *   name?: string,
 *   dim: number,
 *   encodeFn: (text: string) => Promise<Float32Array | number[]>,
 *   batchFn?: (texts: string[]) => Promise<(Float32Array | number[])[]>,
 *   cacheSize?: number
 * }} options
 * @returns {import("../types").Embedder}
 */
export function createTransformerEmbedder(options) {
  if (!options || typeof options.encodeFn !== "function") {
    throw new Error("createTransformerEmbedder: encodeFn is required");
  }
  if (!Number.isFinite(options.dim) || options.dim <= 0) {
    throw new Error("createTransformerEmbedder: dim must be a positive number");
  }
  const cacheSize = options.cacheSize != null ? options.cacheSize : 256;
  const cache = new Map();

  function cachePut(key, value) {
    if (cache.has(key)) cache.delete(key);
    cache.set(key, value);
    while (cache.size > cacheSize) {
      const firstKey = cache.keys().next().value;
      cache.delete(firstKey);
    }
  }

  async function encode(text) {
    const key = String(text || "");
    if (cache.has(key)) {
      const v = cache.get(key);
      cache.delete(key);
      cache.set(key, v);
      return v;
    }
    const v = await options.encodeFn(key);
    cachePut(key, v);
    return v;
  }

  async function batch(texts) {
    if (options.batchFn) {
      const out = await options.batchFn(texts);
      texts.forEach((t, i) => cachePut(String(t || ""), out[i]));
      return out;
    }
    return Promise.all(texts.map((t) => encode(t)));
  }

  return {
    name: options.name || "transformer",
    dim: options.dim,
    encode,
    batch,
    sim(a, b) {
      return cosineSimilarity(a, b);
    },
  };
}

export default createTransformerEmbedder;
