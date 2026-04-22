import { useEffect, useState } from "react";
import { useMemory } from "./MemoryProvider";

// Streams the metrics collector's snapshot into a React component. Useful for
// debug panels and real-time dashboards.

export function useMemoryMetrics() {
  const manager = useMemory();
  const [snapshot, setSnapshot] = useState(() => manager.metrics.snapshot());

  useEffect(() => {
    const off = manager.metrics.subscribe(setSnapshot);
    return () => off();
  }, [manager]);

  return snapshot;
}
