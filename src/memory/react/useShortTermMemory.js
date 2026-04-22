import { useCallback, useEffect, useState } from "react";
import { useMemory } from "./MemoryProvider";

// Subscribes to STM lifecycle events and re-renders with the current items.
// Returns the items plus imperative helpers.

export function useShortTermMemory() {
  const manager = useMemory();
  const [items, setItems] = useState(() => manager.stm.window());

  useEffect(() => {
    const refresh = () => setItems(manager.stm.window());
    refresh();
    const offs = [
      manager.on("stm:add", refresh),
      manager.on("stm:evict", refresh),
      manager.on("stm:boost", refresh),
      manager.on("stm:clear", refresh),
    ];
    return () => {
      for (const off of offs) off();
    };
  }, [manager]);

  const add = useCallback(
    (entry) => manager.stm.add(entry),
    [manager]
  );
  const boost = useCallback(
    (id, delta) => manager.stm.boost(id, delta),
    [manager]
  );
  const clear = useCallback(() => manager.stm.clear(), [manager]);

  return { items, add, boost, clear };
}
