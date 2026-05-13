import { useEffect, useRef, useState } from 'react';
import type { VirtualItem } from './useVirtualList';

export interface UseWindowVirtualizerArgs {
  itemCount: number;
  itemHeight: number;
  overscan?: number;
}

export interface UseWindowVirtualizerResult {
  listRef: React.RefObject<HTMLDivElement>;
  virtualItems: VirtualItem[];
  totalHeight: number;
  startIndex: number;
  endIndex: number;
}

/**
 * Window-based virtualizer. Tracks window.scrollY, accounting for the
 * list's offsetTop so items above the fold don't count.
 */
export default function useWindowVirtualizer({
  itemCount,
  itemHeight,
  overscan = 3,
}: UseWindowVirtualizerArgs): UseWindowVirtualizerResult {
  const listRef = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [listOffsetTop, setListOffsetTop] = useState(0);
  const [viewportHeight, setViewportHeight] = useState(
    typeof window !== 'undefined' ? window.innerHeight : 800,
  );

  useEffect(() => {
    const measureOffset = () => {
      if (listRef.current) {
        setListOffsetTop(listRef.current.getBoundingClientRect().top + window.scrollY);
      }
    };
    measureOffset();
    window.addEventListener('resize', measureOffset);
    return () => window.removeEventListener('resize', measureOffset);
  }, []);

  useEffect(() => {
    const onScroll = () => setScrollTop(window.scrollY);
    const onResize = () => setViewportHeight(window.innerHeight);

    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onResize);

    return () => {
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', onResize);
    };
  }, []);

  const totalHeight = itemCount * itemHeight;
  const scrollRelative = Math.max(0, scrollTop - listOffsetTop);

  const startIndex = Math.max(0, Math.floor(scrollRelative / itemHeight) - overscan);
  const endIndex = Math.min(
    itemCount - 1,
    Math.ceil((scrollRelative + viewportHeight) / itemHeight) - 1 + overscan,
  );

  const virtualItems: VirtualItem[] = [];
  for (let i = startIndex; i <= endIndex; i++) {
    virtualItems.push({ index: i, offsetTop: i * itemHeight });
  }

  return { listRef, virtualItems, totalHeight, startIndex, endIndex };
}
