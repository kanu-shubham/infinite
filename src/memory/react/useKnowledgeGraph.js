import { useCallback, useEffect, useState } from "react";
import { useMemory } from "./MemoryProvider";

// Read-heavy hook around the knowledge graph. Exposes the current size plus
// memoized lookups; traversal and neighbors are returned as callables so the
// host component can decide when to execute potentially expensive queries.

export function useKnowledgeGraph() {
  const manager = useMemory();
  const [size, setSize] = useState(() => manager.kg.size());

  useEffect(() => {
    const refresh = () => setSize(manager.kg.size());
    refresh();
    const off = manager.on("kg:upsert", refresh);
    return () => off();
  }, [manager]);

  const neighbors = useCallback(
    (id, options) => manager.kg.neighbors(id, options),
    [manager]
  );
  const traverse = useCallback(
    (seeds, options) => manager.kg.traverse(seeds, options),
    [manager]
  );
  const getNode = useCallback((id) => manager.kg.getNode(id), [manager]);

  return { size, neighbors, traverse, getNode, graph: manager.kg };
}
