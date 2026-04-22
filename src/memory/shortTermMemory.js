import { EventEmitter, now, uid } from "./utils";
import { createBagOfWordsEmbedder } from "./embedder/bagOfWordsEmbedder";

// Short-term contextual memory: a bounded sliding-window buffer of the most
// recent interactions/turns. Serves as the "working context" that a reasoner
// sees on every step. Items carry salience so unusually important turns
// survive longer than routine ones when the window is full.
//
// Features in this refactor:
//   - Embedder injection (default: bag-of-words) so features are computed
//     consistently with the rest of the stack.
//   - EventEmitter surface: `stm:add`, `stm:evict`, `stm:boost`, `stm:clear`.
//   - Token-budget aware eviction picks the lowest-salience oldest entry.

const DEFAULTS = {
  capacity: 20,
  maxTokens: 2048,
  salienceFloor: 0,
};

export class ShortTermMemory {
  constructor(options = {}) {
    this.opts = { ...DEFAULTS, ...options };
    this.embedder = options.embedder || createBagOfWordsEmbedder();
    this.emitter = options.emitter || new EventEmitter();
    this.items = [];
  }

  async add(entry) {
    const text = typeof entry === "string" ? entry : entry.text || "";
    const features = await this.embedder.encode(text);
    const item = {
      id: uid("stm"),
      text,
      role: entry.role || "user",
      salience: entry.salience != null ? entry.salience : 1,
      tokens: estimateTokens(text),
      features,
      createdAt: now(),
      meta: entry.meta || {},
    };
    this.items.push(item);
    const evicted = this._evict();
    this.emitter.emit("stm:add", { item });
    if (evicted.length) this.emitter.emit("stm:evict", { items: evicted });
    return item;
  }

  window(limit) {
    if (!limit || limit >= this.items.length) return this.items.slice();
    return this.items.slice(this.items.length - limit);
  }

  // Weighted score combining recency and salience. Useful for prompt assembly
  // when the model's budget is tighter than the buffer capacity.
  rank(limit = this.opts.capacity) {
    const t = now();
    const scored = this.items.map((it) => {
      const ageMs = t - it.createdAt;
      const recency = 1 / (1 + ageMs / 60000); // 1 minute scale
      return { item: it, score: recency * (it.salience + 0.001) };
    });
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, limit).map((s) => s.item);
  }

  boost(id, delta = 1) {
    const it = this.items.find((x) => x.id === id);
    if (it) {
      it.salience += delta;
      this.emitter.emit("stm:boost", { id, salience: it.salience });
    }
    return it;
  }

  clear() {
    this.items = [];
    this.emitter.emit("stm:clear", {});
  }

  size() {
    return this.items.length;
  }

  _evict() {
    const { capacity, maxTokens, salienceFloor } = this.opts;
    const evicted = [];

    const keep = [];
    for (const it of this.items) {
      if (it.salience < salienceFloor) evicted.push(it);
      else keep.push(it);
    }
    this.items = keep;

    while (this.items.length > capacity) {
      evicted.push(this.items.shift());
    }

    let totalTokens = this.items.reduce((s, it) => s + it.tokens, 0);
    while (totalTokens > maxTokens && this.items.length > 1) {
      let evictIdx = 0;
      let evictScore = Infinity;
      for (let i = 0; i < this.items.length - 1; i++) {
        const it = this.items[i];
        const score = it.salience - (this.items.length - i) * 0.01;
        if (score < evictScore) {
          evictScore = score;
          evictIdx = i;
        }
      }
      totalTokens -= this.items[evictIdx].tokens;
      evicted.push(this.items.splice(evictIdx, 1)[0]);
    }
    return evicted;
  }
}

function estimateTokens(text) {
  if (!text) return 0;
  return Math.ceil(text.length / 4) + text.split(/\s+/).filter(Boolean).length;
}

export default ShortTermMemory;
