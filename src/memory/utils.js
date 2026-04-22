// Shared utilities for memory modules. Dependency-free so the library can be
// lifted out of this React app into any JS runtime (browser, Node, worker).

export function now() {
  return Date.now();
}

export function uid(prefix = "m") {
  return `${prefix}_${Math.random().toString(36).slice(2, 10)}_${now().toString(36)}`;
}

// Lightweight bag-of-words feature vector keyed on tokens. Not a semantic
// embedding, but good enough for similarity scoring without external models.
const STOPWORDS = new Set([
  "a", "an", "the", "and", "or", "but", "if", "then", "of", "to", "in",
  "on", "at", "for", "is", "are", "was", "were", "be", "been", "being",
  "it", "its", "this", "that", "these", "those", "with", "as", "by", "from",
  "i", "you", "he", "she", "we", "they", "them", "me", "my", "your", "our"
]);

export function tokenize(text) {
  if (text == null) return [];
  return String(text)
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s'-]/gu, " ")
    .split(/\s+/)
    .filter((t) => t && !STOPWORDS.has(t));
}

export function featureVector(text) {
  const vec = new Map();
  for (const tok of tokenize(text)) {
    vec.set(tok, (vec.get(tok) || 0) + 1);
  }
  return vec;
}

export function cosineSimilarity(a, b) {
  if (!a || !b) return 0;
  if (a instanceof Map && b instanceof Map) {
    if (a.size === 0 || b.size === 0) return 0;
    let dot = 0;
    let na = 0;
    let nb = 0;
    for (const [, v] of a) na += v * v;
    for (const [, v] of b) nb += v * v;
    const [small, large] = a.size <= b.size ? [a, b] : [b, a];
    for (const [k, v] of small) {
      const o = large.get(k);
      if (o) dot += v * o;
    }
    const denom = Math.sqrt(na) * Math.sqrt(nb);
    return denom === 0 ? 0 : dot / denom;
  }
  // Dense vectors (Float32Array / Array / TypedArray).
  const len = Math.min(a.length, b.length);
  if (len === 0) return 0;
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < len; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 0 : dot / denom;
}

// Exponential decay applied to a recall strength value given elapsed ms and a
// half-life. Used by episodic memory to model forgetting.
export function decayStrength(strength, elapsedMs, halfLifeMs) {
  if (halfLifeMs <= 0) return strength;
  const k = Math.log(2) / halfLifeMs;
  return strength * Math.exp(-k * elapsedMs);
}

export function clamp(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}

// Tiny event emitter. Subscribers are isolated from each other — a thrown
// listener doesn't block others or the emit caller.
export class EventEmitter {
  constructor() {
    this._listeners = new Map();
  }

  on(event, fn) {
    let set = this._listeners.get(event);
    if (!set) {
      set = new Set();
      this._listeners.set(event, set);
    }
    set.add(fn);
    return () => this.off(event, fn);
  }

  off(event, fn) {
    const set = this._listeners.get(event);
    if (set) set.delete(fn);
  }

  emit(event, payload) {
    const set = this._listeners.get(event);
    if (!set) return;
    for (const fn of set) {
      try {
        fn(payload);
      } catch (err) {
        if (typeof console !== "undefined") {
          console.error(`[memory] listener threw on "${event}"`, err);
        }
      }
    }
  }

  removeAll() {
    this._listeners.clear();
  }
}

// Serializes async operations per-key. Used to prevent concurrent writes to
// the same storage key from clobbering each other, and to dedupe in-flight
// reads. The queue keeps only the last pending write per key to coalesce
// bursts of writes to the same resource.
export class SingleFlightQueue {
  constructor() {
    this._chain = new Map();
    this._pending = new Map();
  }

  run(key, fn) {
    const prev = this._chain.get(key) || Promise.resolve();
    const next = prev.catch(() => {}).then(() => fn());
    this._chain.set(
      key,
      next.catch(() => {})
    );
    next.finally(() => {
      if (this._chain.get(key) === next) this._chain.delete(key);
    });
    return next;
  }

  // Dedupe in-flight calls by key. The first caller wins; subsequent callers
  // with the same key await the original promise.
  dedupe(key, fn) {
    const existing = this._pending.get(key);
    if (existing) return existing;
    const p = Promise.resolve().then(fn);
    this._pending.set(key, p);
    p.finally(() => {
      if (this._pending.get(key) === p) this._pending.delete(key);
    });
    return p;
  }
}
