import { EventEmitter, SingleFlightQueue, decayStrength, now, uid } from "./utils";
import { createBagOfWordsEmbedder } from "./embedder/bagOfWordsEmbedder";
import { MemoryAdapter } from "./storage/memoryAdapter";

// Long-term episodic memory: persistent store of discrete episodes
// (experiences, observations, completed tasks). Recall is similarity-based
// over the injected embedder, with a forgetting curve and reinforcement on
// access so frequently-useful episodes stay retrievable.
//
// Features in this refactor:
//   - Pluggable `Embedder` (bag-of-words default; any dense encoder works).
//   - Pluggable `StorageAdapter` (in-memory default; localStorage via adapter).
//   - EventEmitter surface: `ltm:remember`, `ltm:recall`, `ltm:forget`.
//   - SingleFlightQueue coalesces concurrent writes to the store payload.

const DEFAULTS = {
  halfLifeMs: 1000 * 60 * 60 * 24 * 14, // two weeks
  reinforceOnRecall: 0.5,
  minStrength: 0.01,
  maxEpisodes: 5000,
  storageKey: "ltm.episodes",
};

export class LongTermMemory {
  constructor(options = {}) {
    this.opts = { ...DEFAULTS, ...options };
    this.embedder = options.embedder || createBagOfWordsEmbedder();
    this.storage = options.storage || new MemoryAdapter();
    this.emitter = options.emitter || new EventEmitter();
    this.queue = options.queue || new SingleFlightQueue();
    this.episodes = new Map();
    this._loaded = false;
  }

  async init() {
    if (this._loaded) return;
    await this.storage.open();
    const raw = await this.storage.get(this.opts.storageKey);
    if (Array.isArray(raw)) {
      for (const row of raw) {
        const features = row.features
          ? rehydrateFeatures(row.features)
          : await this.embedder.encode(row.text);
        this.episodes.set(row.id, { ...row, features });
      }
    }
    this._loaded = true;
  }

  async remember(episode) {
    if (!this._loaded) await this.init();
    const text =
      typeof episode === "string"
        ? episode
        : episode.text || episode.summary || "";
    const features = await this.embedder.encode(text);
    const ep = {
      id: (episode && episode.id) || uid("ep"),
      text,
      tags: Array.isArray(episode && episode.tags) ? [...episode.tags] : [],
      features,
      strength: episode && episode.strength != null ? episode.strength : 1,
      createdAt: (episode && episode.createdAt) || now(),
      lastAccessed: now(),
      accessCount: 0,
      meta: (episode && episode.meta) || {},
    };
    this.episodes.set(ep.id, ep);
    this._enforceCap();
    this.emitter.emit("ltm:remember", { episode: ep });
    this._persist();
    return ep;
  }

  async forget(id) {
    if (!this._loaded) await this.init();
    const ep = this.episodes.get(id);
    const removed = this.episodes.delete(id);
    if (removed) {
      this.emitter.emit("ltm:forget", { id, episode: ep });
      this._persist();
    }
    return removed;
  }

  // Returns episodes ranked by a combined similarity + current strength score.
  // Accessing an episode reinforces it (Ebbinghaus-style spaced repetition).
  async recall(query, { limit = 5, tag = null, threshold = 0.05 } = {}) {
    if (!this._loaded) await this.init();
    const qvec = await this.embedder.encode(query);
    const t = now();
    const hits = [];
    for (const ep of this.episodes.values()) {
      if (tag && !ep.tags.includes(tag)) continue;
      const sim = this.embedder.sim(qvec, ep.features);
      if (sim < threshold) continue;
      const currentStrength = decayStrength(
        ep.strength,
        t - ep.lastAccessed,
        this.opts.halfLifeMs
      );
      if (currentStrength < this.opts.minStrength) continue;
      const score = sim * (0.5 + currentStrength);
      hits.push({
        episode: ep,
        score,
        similarity: sim,
        strength: currentStrength,
      });
    }
    hits.sort((a, b) => b.score - a.score);
    const top = hits.slice(0, limit);
    for (const h of top) this._reinforce(h.episode);
    if (top.length) this._persist();
    this.emitter.emit("ltm:recall", { query, hits: top });
    return top;
  }

  async get(id) {
    if (!this._loaded) await this.init();
    const ep = this.episodes.get(id);
    if (ep) {
      this._reinforce(ep);
      this._persist();
    }
    return ep;
  }

  all() {
    return Array.from(this.episodes.values());
  }

  size() {
    return this.episodes.size;
  }

  // Apply passive decay. Call periodically (or let recall handle it lazily).
  async consolidate() {
    if (!this._loaded) await this.init();
    const t = now();
    let dropped = 0;
    for (const [id, ep] of this.episodes) {
      const s = decayStrength(
        ep.strength,
        t - ep.lastAccessed,
        this.opts.halfLifeMs
      );
      if (s < this.opts.minStrength) {
        this.episodes.delete(id);
        dropped++;
      } else {
        ep.strength = s;
        ep.lastAccessed = t;
      }
    }
    if (dropped) {
      this.emitter.emit("decay:run", { dropped });
      this._persist();
    }
    return dropped;
  }

  _reinforce(ep) {
    ep.strength = Math.min(10, ep.strength + this.opts.reinforceOnRecall);
    ep.lastAccessed = now();
    ep.accessCount += 1;
  }

  _enforceCap() {
    if (this.episodes.size <= this.opts.maxEpisodes) return;
    const t = now();
    const entries = Array.from(this.episodes.values()).map((ep) => ({
      ep,
      s: decayStrength(ep.strength, t - ep.lastAccessed, this.opts.halfLifeMs),
    }));
    entries.sort((a, b) => a.s - b.s);
    while (this.episodes.size > this.opts.maxEpisodes && entries.length) {
      const victim = entries.shift();
      this.episodes.delete(victim.ep.id);
    }
  }

  // Serialize through the queue so overlapping remembers/forgets coalesce
  // into ordered writes. We intentionally only persist the latest snapshot.
  _persist() {
    this.queue.run(this.opts.storageKey, async () => {
      const payload = Array.from(this.episodes.values()).map(serializeEpisode);
      await this.storage.set(this.opts.storageKey, payload);
    });
  }
}

function serializeEpisode(ep) {
  return {
    id: ep.id,
    text: ep.text,
    tags: ep.tags,
    strength: ep.strength,
    createdAt: ep.createdAt,
    lastAccessed: ep.lastAccessed,
    accessCount: ep.accessCount,
    meta: ep.meta,
    features: serializeFeatures(ep.features),
  };
}

function serializeFeatures(v) {
  if (v instanceof Map) return { kind: "map", entries: Array.from(v.entries()) };
  if (v instanceof Float32Array) return { kind: "f32", data: Array.from(v) };
  if (Array.isArray(v)) return { kind: "arr", data: v };
  return null;
}

function rehydrateFeatures(v) {
  if (!v || typeof v !== "object") return new Map();
  if (v.kind === "map") return new Map(v.entries);
  if (v.kind === "f32") return Float32Array.from(v.data);
  if (v.kind === "arr") return v.data.slice();
  return new Map();
}

export default LongTermMemory;
