import { useCallback, useEffect, useRef } from 'react';

export interface UseInfiniteScrollOptions {
  enabled?: boolean;
}

export default function useInfiniteScroll(
  onLoadMore: () => void,
  { enabled = true }: UseInfiniteScrollOptions = {},
): (node: Element | null) => void {
  const observerRef = useRef<IntersectionObserver | null>(null);

  const cleanup = useCallback(() => {
    if (observerRef.current) {
      observerRef.current.disconnect();
      observerRef.current = null;
    }
  }, []);

  useEffect(() => cleanup, [cleanup]);

  const sentinelRef = useCallback(
    (node: Element | null) => {
      cleanup();
      if (!node || !enabled) return;

      observerRef.current = new IntersectionObserver(
        ([entry]) => {
          if (entry.isIntersecting) onLoadMore();
        },
        { rootMargin: '200px' },
      );

      observerRef.current.observe(node);
    },
    [enabled, onLoadMore, cleanup],
  );

  return sentinelRef;
}
