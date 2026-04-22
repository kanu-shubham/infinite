import { useCallback, useEffect, useState } from "react";
import { useMemory } from "./MemoryProvider";

// Low-level access to the adaptive retriever for debug panels and
// administrative overrides. Streams bandit state updates and exposes manual
// weight / feedback controls.

export function useAdaptiveRetrieval() {
  const manager = useMemory();
  const [stats, setStats] = useState(() => manager.retriever.stats());

  useEffect(() => {
    const refresh = () => setStats(manager.retriever.stats());
    refresh();
    const offs = [
      manager.on("retrieval:done", refresh),
      manager.on("feedback:applied", refresh),
    ];
    return () => {
      for (const off of offs) off();
    };
  }, [manager]);

  const feedback = useCallback(
    (resultId, reward) => manager.feedback({ resultId, reward }),
    [manager]
  );
  const resetBandit = useCallback(() => {
    manager.retriever.reset();
    setStats(manager.retriever.stats());
  }, [manager]);

  return {
    weights: stats.weights,
    bandit: stats.bandit,
    queries: stats.queries,
    feedback,
    resetBandit,
    retriever: manager.retriever,
  };
}
