import { useCallback, useRef, useEffect } from "react";

export default function useInfiniteScroll(onLoadMore, { enabled = true } = {}) {
  const observerRef = useRef(null);

  const cleanup = useCallback(() => {
    if (observerRef.current) {
      observerRef.current.disconnect();
      observerRef.current = null;
    }
  }, []);

  useEffect(() => cleanup, [cleanup]);

  const sentinelRef = useCallback(
    (node) => {
      cleanup();

      if (!node || !enabled) return;

      observerRef.current = new IntersectionObserver(
        ([entry]) => {
          if (entry.isIntersecting) {
            onLoadMore();
          }
        },
        { rootMargin: "200px" }
      );

      observerRef.current.observe(node);
    },
    [enabled, onLoadMore, cleanup]
  );

  return sentinelRef;
}
