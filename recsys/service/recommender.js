// Recommender service.
//
// Public-facing service that wires the model layers together with the state
// stores. This is the boundary the API layer talks to. It exposes two
// methods:
//
//   recommend(userId, opts)  -> { items, hasMore, page, pageSize, debug }
//   ingestEvent(event)       -> updated profile (+ fan-out to event log)
//
// Everything below this method is pure model code; everything above is
// transport/API.

const { generateCandidates } = require("../model/candidateGenerators");
const { rank, explain, DEFAULT_WEIGHTS } = require("../model/ranker");
const { rankWithLearned, explainLearned } = require("../model/learnedRanker");
const { diversify, applyHardFilters } = require("../model/reranker");
const { applyEvent, isColdStart } = require("../model/userProfile");

class Recommender {
  constructor({ catalog, profileStore, eventLog, weights = DEFAULT_WEIGHTS, learnedModel = null }) {
    this.catalog = catalog;
    this.profileStore = profileStore;
    this.eventLog = eventLog;
    this.weights = weights;
    this.learnedModel = learnedModel;

    // Wire event-log -> profile updates so writes from any path keep state
    // consistent. The profile store is updated synchronously so the next
    // `recommend` call sees the new state.
    this.eventLog.subscribe((evt) => this._applyEventToProfile(evt));
  }

  _applyEventToProfile(event) {
    const current = this.profileStore.get(event.userId);
    const next = applyEvent(current, event, this.catalog);
    this.profileStore.put(event.userId, next);
  }

  // Public — called by the API layer.
  ingestEvent(event) {
    if (!event || !event.userId || !event.videoId || !event.type) {
      throw new Error("event must include userId, videoId, type");
    }
    if (!["watch", "like", "skip"].includes(event.type)) {
      throw new Error(`unknown event type: ${event.type}`);
    }
    return this.eventLog.append(event);
  }

  // Public — called by the API layer.
  recommend(userId, { page = 1, pageSize = 8, blocklist = [] } = {}) {
    const profile = this.profileStore.get(userId);

    // 1. Candidate generation. Learned MF retrieval replaces the heuristic
    //    content/collab generators when an artifact is loaded.
    const rawCandidates = generateCandidates(profile, this.catalog, {
      learnedModel: this.learnedModel,
    });

    // 2. Hard filters.
    const excludeIds = new Set([
      ...profile.watchHistory,
      ...profile.skipped,
    ]);
    const filtered = applyHardFilters(rawCandidates, {
      excludeIds,
      blocklist: new Set(blocklist),
    });

    // 3. Rank. Learned logistic regression model takes over when present.
    const ranked = this.learnedModel
      ? rankWithLearned(filtered, profile, this.catalog, this.learnedModel)
      : rank(filtered, profile, this.weights);
    const explainFn = this.learnedModel ? explainLearned : explain;

    // 4. Re-rank for diversity. Build the slate large enough for the
    //    requested page.
    const slateSize = Math.min(ranked.length, page * pageSize + 8);
    const slate = diversify(ranked, slateSize);

    const start = (page - 1) * pageSize;
    const items = slate.slice(start, start + pageSize).map((s) => ({
      videoId: s.video.id,
      title: s.video.title,
      channelId: s.video.channelId,
      channelName: s.video.channelName,
      tags: s.video.tags,
      durationSec: s.video.durationSec,
      uploadedAt: s.video.uploadedAt,
      source: s.source,
      finalScore: s.finalScore,
      rankerScore: s.rankerScore,
      reasons: explainFn(s).map((e) => e.feature),
    }));

    return {
      items,
      hasMore: slate.length > start + pageSize,
      page,
      pageSize,
      debug: {
        userId,
        coldStart: isColdStart(profile),
        modelKind: this.learnedModel ? "learned" : "heuristic",
        candidatePoolSize: rawCandidates.length,
        afterFilter: filtered.length,
        slateSize: slate.length,
      },
    };
  }
}

module.exports = { Recommender };
