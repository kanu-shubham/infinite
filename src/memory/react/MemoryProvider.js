import React, {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useRef,
} from "react";
import { createMemoryManager } from "../MemoryManager";

const MemoryContext = createContext(null);

// Mounts (and disposes) a MemoryManager for the subtree. Accepts either a
// prebuilt manager or a config object to lazily build one. Flushes pending
// work on page unload so localStorage writes land before tab teardown.

export function MemoryProvider({
  manager: externalManager,
  config,
  children,
}) {
  const ownedRef = useRef(null);

  const manager = useMemo(() => {
    if (externalManager) return externalManager;
    const owned = createMemoryManager(config || {});
    ownedRef.current = owned;
    return owned;
  }, [externalManager, config]);

  useEffect(() => {
    let cancelled = false;
    manager.init().catch((err) => {
      if (!cancelled && typeof console !== "undefined") {
        console.error("[memory] init failed", err);
      }
    });
    const onUnload = () => {
      manager.consolidate().catch(() => {});
    };
    if (typeof window !== "undefined") {
      window.addEventListener("beforeunload", onUnload);
    }
    return () => {
      cancelled = true;
      if (typeof window !== "undefined") {
        window.removeEventListener("beforeunload", onUnload);
      }
      if (ownedRef.current === manager) {
        manager.dispose().catch(() => {});
      }
    };
  }, [manager]);

  return (
    <MemoryContext.Provider value={manager}>{children}</MemoryContext.Provider>
  );
}

export function useMemory() {
  const mgr = useContext(MemoryContext);
  if (!mgr) {
    throw new Error("useMemory: component must be inside <MemoryProvider>");
  }
  return mgr;
}
