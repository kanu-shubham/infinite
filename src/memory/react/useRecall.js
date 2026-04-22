import { useCallback, useEffect, useRef, useState } from "react";
import { useMemory } from "./MemoryProvider";

// Debounced adaptive recall. Re-runs whenever `query` stabilizes, and exposes
// explain() + feedback() to surface and tune signal fusion in the UI.

export function useRecall(query, { k = 5, debounceMs = 250, enabled = true } = {}) {
  const manager = useMemory();
  const [state, setState] = useState({
    results: [],
    weights: null,
    latencyMs: 0,
    isLoading: false,
  });
  const runId = useRef(0);

  useEffect(() => {
    if (!enabled) return undefined;
    const q = String(query || "").trim();
    if (!q) {
      setState((s) => ({ ...s, results: [], isLoading: false }));
      return undefined;
    }
    const id = ++runId.current;
    setState((s) => ({ ...s, isLoading: true }));
    const handle = setTimeout(async () => {
      try {
        const out = await manager.query(q, { k });
        if (id !== runId.current) return;
        setState({
          results: out.results,
          weights: out.weights,
          latencyMs: out.latencyMs,
          isLoading: false,
        });
      } catch (err) {
        if (id !== runId.current) return;
        setState((s) => ({ ...s, isLoading: false }));
        if (typeof console !== "undefined") console.error("[memory] recall", err);
      }
    }, debounceMs);
    return () => clearTimeout(handle);
  }, [manager, query, k, debounceMs, enabled]);

  const feedback = useCallback(
    (resultId, reward) => manager.feedback({ resultId, reward }),
    [manager]
  );
  const explain = useCallback(
    (resultId) => manager.retriever.explain(resultId),
    [manager]
  );

  return { ...state, feedback, explain };
}
