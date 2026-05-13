/**
 * useVariableVirtualList — Dynamic / variable-height virtualizer.
 *
 * Drop-in replacement for useVirtualList when item heights are unknown
 * or differ between items. NOT currently wired up (kept for swap-in use).
 *
 * Mechanism:
 *   1. Every rendered item gets a ResizeObserver via getMeasureRef(index).
 *   2. First measurement stores the real height in itemHeightsRef.
 *   3. offsets[i] is the prefix-sum (top edge) of item i.
 *   4. Binary search on offsets locates the first visible item in O(log n).
 *   5. Height changes rebuild offsets from the changed index forward.
 */

import { useCallback, useEffect, useReducer, useRef, useState } from 'react';

export interface VariableVirtualItem {
  index: number;
  offsetTop: number;
  size: number;
}

export interface UseVariableVirtualListArgs {
  itemCount: number;
  estimatedItemHeight?: number;
  overscan?: number;
}

export interface UseVariableVirtualListResult {
  containerRef: React.RefObject<HTMLDivElement>;
  virtualItems: VariableVirtualItem[];
  totalHeight: number;
  getMeasureRef: (index: number) => (node: Element | null) => void;
  startIndex: number;
  endIndex: number;
}

function buildOffsets(heights: number[]): number[] {
  const offsets = new Array<number>(heights.length + 1);
  offsets[0] = 0;
  for (let i = 0; i < heights.length; i++) {
    offsets[i + 1] = offsets[i] + heights[i];
  }
  return offsets;
}

/** Binary search: largest index whose offset <= target. */
function binarySearchOffset(offsets: number[], target: number): number {
  let lo = 0;
  let hi = offsets.length - 2;
  while (lo < hi) {
    const mid = Math.floor((lo + hi + 1) / 2);
    if (offsets[mid] <= target) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

export default function useVariableVirtualList({
  itemCount,
  estimatedItemHeight = 200,
  overscan = 3,
}: UseVariableVirtualListArgs): UseVariableVirtualListResult {
  const containerRef = useRef<HTMLDivElement>(null);
  const itemHeightsRef = useRef<number[]>(new Array(itemCount).fill(estimatedItemHeight));
  const offsetsRef = useRef<number[]>(buildOffsets(itemHeightsRef.current));
  const observersRef = useRef<Map<number, ResizeObserver>>(new Map());

  const [scrollTop, setScrollTop] = useState(0);
  const [containerHeight, setContainerHeight] = useState(500);
  const [, forceUpdate] = useReducer((x: number) => x + 1, 0);

  const recalcOffsets = useCallback(
    (fromIndex: number) => {
      const heights = itemHeightsRef.current;
      const offsets = offsetsRef.current;
      offsets[fromIndex] = fromIndex === 0 ? 0 : offsets[fromIndex];
      for (let i = fromIndex; i < itemCount; i++) {
        offsets[i + 1] = offsets[i] + heights[i];
      }
    },
    [itemCount],
  );

  useEffect(() => {
    itemHeightsRef.current = new Array(itemCount).fill(estimatedItemHeight);
    offsetsRef.current = buildOffsets(itemHeightsRef.current);
    observersRef.current.forEach((ro) => ro.disconnect());
    observersRef.current.clear();
    forceUpdate();
  }, [itemCount, estimatedItemHeight]);

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

  useEffect(() => {
    const observers = observersRef.current;
    return () => observers.forEach((ro) => ro.disconnect());
  }, []);

  const getMeasureRef = useCallback(
    (index: number) => (node: Element | null) => {
      if (observersRef.current.has(index)) {
        observersRef.current.get(index)?.disconnect();
        observersRef.current.delete(index);
      }
      if (!node) return;

      const ro = new ResizeObserver(([entry]) => {
        const measuredHeight = entry.contentRect.height;
        if (itemHeightsRef.current[index] !== measuredHeight) {
          itemHeightsRef.current[index] = measuredHeight;
          recalcOffsets(index);
          forceUpdate();
        }
      });

      ro.observe(node);
      observersRef.current.set(index, ro);
    },
    [recalcOffsets],
  );

  const offsets = offsetsRef.current;
  const totalHeight = offsets[itemCount] ?? itemCount * estimatedItemHeight;

  const startIndex = Math.max(0, binarySearchOffset(offsets, scrollTop) - overscan);

  let endIndex = startIndex;
  while (endIndex < itemCount - 1 && offsets[endIndex + 1] < scrollTop + containerHeight) {
    endIndex++;
  }
  endIndex = Math.min(itemCount - 1, endIndex + overscan);

  const virtualItems: VariableVirtualItem[] = [];
  for (let i = startIndex; i <= endIndex; i++) {
    virtualItems.push({
      index: i,
      offsetTop: offsets[i],
      size: itemHeightsRef.current[i],
    });
  }

  return {
    containerRef,
    virtualItems,
    totalHeight,
    getMeasureRef,
    startIndex,
    endIndex,
  };
}
