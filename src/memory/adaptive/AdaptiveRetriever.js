import { EventEmitter, now } from "../utils";
import { createEpsilonGreedyBandit } from "./bandit";
import { normalizeKG, normalizeLTM, normalizeRecency, normalizeSTM } from "./signals";

// Fuses signals from short-term memory, long-term episodic memory, and the
// knowledge graph into a single ranked result list. Fusion weights are held
// by a bandit that learns from `feedback()` calls so the retriever adapts to
// what the current user actually finds useful.

const ARMS = ["stm", "ltm", "kg", "recency"];
const DEFAULTS = {
  topK: 10,
  warmupQueries: 5,
  warmupWeights: { stm: 0.3, ltm: 0.4, kg: 0.2, recency: 0.1 },
  stmLimit: 20,
  ltmLimit: 10,
  kgTraverseDepth: 2,
  kgBudgetMs: 30,
};

export class AdaptiveRetriever {
  constructor({ stm, ltm, kg, emitter, bandit, ...options } = {}) {
    if (!stm || !ltm || !kg) {
      throw new Error("AdaptiveRetriever: stm, ltm, kg are required");
    }
    this.stm = stm;
    this.ltm = ltm;
    this.kg = kg;
    this.opts = { ...DEFAULTS, ...options };
    this.emitter = emitter || new EventEmitter();
    this.bandit = bandit || createEpsilonGreedyBandit(ARMS);
    this._queryCount = 0;
    this._lastExplanations = new Map();
  }

  async query(text, { k = this.opts.topK, seedIds = [] } = {}) {
    const start = now();
    this._queryCount += 1;

    const stmItems = this.stm.rank(this.opts.stmLimit);
    const [ltmHits, kgHits] = await Promise.all([
      this.ltm.recall(text, { limit: this.opts.ltmLimit }),
      Promise.resolve(
        kgSeeds(this.kg, text, seedIds).length
          ? this.kg.traverse(kgSeeds(this.kg, text, seedIds), {
              maxDepth: this.opts.kgTraverseDepth,
              budgetMs: this.opts.kgBudgetMs,
              limit: this.opts.ltmLimit,
            })
          : []
      ),
    ]);

    const stmSignals = normalizeSTM(stmItems);
    const ltmSignals = normalizeLTM(ltmHits);
    const kgSignals = normalizeKG(kgHits);
    const recencySignals = normalizeRecency(stmItems);

    const weights = this._weights();
    const merged = new Map();
    for (const signals of [stmSignals, ltmSignals, kgSignals, recencySignals]) {
      for (const sig of signals) {
        const w = weights[sig.source] || 0;
        const contrib = w * sig.score;
        const entry = merged.get(sig.id) || {
          id: sig.id,
          score: 0,
          signals: [],
          payload: sig.payload,
        };
        entry.score += contrib;
        entry.signals.push(sig);
        if (!entry.payload) entry.payload = sig.payload;
        merged.set(sig.id, entry);
      }
    }

    const results = Array.from(merged.values())
      .sort((a, b) => b.score - a.score)
      .slice(0, k);

    this._lastExplanations.clear();
    for (const r of results) this._lastExplanations.set(r.id, r);

    const latencyMs = now() - start;
    this.emitter.emit("retrieval:done", {
      query: text,
      k,
      latencyMs,
      results,
      weights,
    });
    return { results, weights, latencyMs };
  }

  feedback({ resultId, reward }) {
    const r = this._lastExplanations.get(resultId);
    if (!r) return null;
    const dominant = dominantSource(r.signals);
    if (!dominant) return null;
    this.bandit.update(dominant, clampReward(reward));
    this.emitter.emit("feedback:applied", {
      resultId,
      reward,
      arm: dominant,
      snapshot: this.bandit.snapshot(),
    });
    return { arm: dominant, weights: this.bandit.weights() };
  }

  explain(resultId) {
    return this._lastExplanations.get(resultId) || null;
  }

  weights() {
    return this._weights();
  }

  stats() {
    return {
      queries: this._queryCount,
      bandit: this.bandit.snapshot(),
      weights: this._weights(),
    };
  }

  reset() {
    this._queryCount = 0;
    this.bandit.reset();
    this._lastExplanations.clear();
  }

  _weights() {
    if (this._queryCount <= this.opts.warmupQueries) {
      return { ...this.opts.warmupWeights };
    }
    return this.bandit.weights();
  }
}

function kgSeeds(kg, text, explicit) {
  const seeds = new Set(explicit);
  const tokens = String(text || "")
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean);
  for (const tok of tokens) {
    if (kg.getNode(`amenity:${tok}`)) seeds.add(`amenity:${tok}`);
    if (kg.getNode(`location:${tok}`)) seeds.add(`location:${tok}`);
  }
  return Array.from(seeds);
}

function dominantSource(signals) {
  if (!signals || !signals.length) return null;
  let best = null;
  let bestScore = -Infinity;
  for (const s of signals) {
    if (s.score > bestScore) {
      bestScore = s.score;
      best = s.source;
    }
  }
  return best;
}

function clampReward(r) {
  if (!Number.isFinite(r)) return 0;
  if (r > 1) return 1;
  if (r < -1) return -1;
  return r;
}

export default AdaptiveRetriever;
