import { useCallback, useEffect, useRef, useState } from 'react';

export interface VirtualItem {
  index: number;
  offsetTop: number;
}

export interface UseVirtualListArgs {
  itemCount: number;
  itemHeight: number;
  overscan?: number;
}

export interface UseVirtualListResult {
  containerRef: React.RefObject<HTMLDivElement>;
  virtualItems: VirtualItem[];
  totalHeight: number;
  startIndex: number;
  endIndex: number;
}

/**
 * Container-based virtualizer. Only renders items inside the container's
 * visible viewport plus `overscan` rows above and below.
 */
export default function useVirtualList({
  itemCount,
  itemHeight,
  overscan = 3,
}: UseVirtualListArgs): UseVirtualListResult {
  const containerRef = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [containerHeight, setContainerHeight] = useState(500);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return undefined;

    setContainerHeight(container.clientHeight);

    const onScroll = () => setScrollTop(container.scrollTop);
    container.addEventListener('scroll', onScroll, { passive: true });

    const ro = new ResizeObserver(([entry]) => {
      setContainerHeight(entry.contentRect.height);
    });
    ro.observe(container);

    return () => {
      container.removeEventListener('scroll', onScroll);
      ro.disconnect();
    };
  }, []);

  const totalHeight = itemCount * itemHeight;

  const startIndex = Math.max(0, Math.floor(scrollTop / itemHeight) - overscan);
  const endIndex = Math.min(
    itemCount - 1,
    Math.ceil((scrollTop + containerHeight) / itemHeight) - 1 + overscan,
  );

  const virtualItems: VirtualItem[] = [];
  for (let i = startIndex; i <= endIndex; i++) {
    virtualItems.push({ index: i, offsetTop: i * itemHeight });
  }

  return { containerRef, virtualItems, totalHeight, startIndex, endIndex };
}
