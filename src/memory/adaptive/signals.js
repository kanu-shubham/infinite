// Per-source normalizers. Each takes raw hits from a subsystem and returns
// a uniform list of signals scored in [0, 1]. Keeping normalization here
// rather than inside each subsystem lets the retriever tune fusion without
// round-tripping through STM/LTM/KG internals.

import { now } from "../utils";

export function normalizeSTM(items) {
  if (!items.length) return [];
  const t = now();
  let max = 0;
  const scored = items.map((it) => {
    const ageMs = t - it.createdAt;
    const recency = 1 / (1 + ageMs / 60000);
    const score = recency * (it.salience + 0.001);
    if (score > max) max = score;
    return { it, score };
  });
  return scored.map(({ it, score }) => ({
    source: "stm",
    id: it.id,
    score: max > 0 ? score / max : 0,
    payload: it,
  }));
}

export function normalizeLTM(hits) {
  if (!hits.length) return [];
  const max = hits.reduce((m, h) => Math.max(m, h.score), 0);
  return hits.map((h) => ({
    source: "ltm",
    id: h.episode.id,
    score: max > 0 ? h.score / max : 0,
    payload: h.episode,
    similarity: h.similarity,
  }));
}

export function normalizeKG(nodes) {
  if (!nodes.length) return [];
  const max = nodes.reduce((m, n) => Math.max(m, n.weight), 0);
  return nodes.map((n) => ({
    source: "kg",
    id: n.node.id,
    score: max > 0 ? n.weight / max : 0,
    payload: n.node,
  }));
}

// A simple recency signal over any items exposing a `createdAt`. Useful as a
// standalone arm so the bandit can privilege "fresh" items independently of
// STM membership.
export function normalizeRecency(items) {
  if (!items.length) return [];
  const t = now();
  const ages = items.map((it) => t - (it.createdAt || t));
  const maxAge = Math.max(1, ...ages);
  return items.map((it, i) => ({
    source: "recency",
    id: it.id,
    score: 1 - ages[i] / maxAge,
    payload: it,
  }));
}
