/**
 * useVariableVirtualList — Dynamic / variable-height virtualizer
 * ─────────────────────────────────────────────────────────────────
 * NOT currently wired up. Drop-in replacement for useVirtualList when
 * item heights are unknown or differ between items (e.g. hotel cards
 * with varying amounts of amenities, multi-line names, etc.).
 *
 * HOW TO SWAP IN:
 *   Replace in VirtualHotelList.js:
 *     import useVirtualList        from "../hooks/useVirtualList";
 *   with:
 *     import useVariableVirtualList from "../hooks/useVariableVirtualList";
 *
 *   Then attach getMeasureRef to each rendered item:
 *     const { containerRef, virtualItems, totalHeight, getMeasureRef } =
 *       useVariableVirtualList({ itemCount: hotels.length });
 *
 *     {virtualItems.map(({ index, offsetTop }) => (
 *       <div
 *         key={hotels[index].id}
 *         ref={getMeasureRef(index)}          // ← extra prop vs fixed-height
 *         style={{ position: "absolute", top: offsetTop, width: "100%" }}
 *       >
 *         <HotelCard hotel={hotels[index]} />
 *       </div>
 *     ))}
 *
 * HOW IT WORKS:
 *   1. Every rendered item gets a ResizeObserver via getMeasureRef(index).
 *   2. When an item renders for the first time its real height is measured
 *      and stored in itemHeightsRef.
 *   3. offsets[i] = sum of all heights before index i (prefix sum array).
 *   4. Binary search on offsets finds the first visible item for a given
 *      scrollTop in O(log n) instead of O(n).
 *   5. When a height changes (dynamic content, images loading) offsets are
 *      rebuilt from that index forward — O(n) worst case but typically fast
 *      because changes happen near the viewport.
 */

import { useState, useEffect, useReducer, useRef, useCallback } from "react";

// ─── Helpers ────────────────────────────────────────────────────────────────

/** Build a prefix-sum offsets array from a heights array.
 *  offsets[i] = top edge of item i.
 *  offsets[itemCount] = total list height.
 */
function buildOffsets(heights) {
  const offsets = new Array(heights.length + 1);
  offsets[0] = 0;
  for (let i = 0; i < heights.length; i++) {
    offsets[i + 1] = offsets[i] + heights[i];
  }
  return offsets;
}

/**
 * Binary search: largest index whose offset <= target.
 * Returns the index of the item that contains scrollTop.
 */
function binarySearchOffset(offsets, target) {
  let lo = 0;
  let hi = offsets.length - 2; // last valid item index
  while (lo < hi) {
    const mid = Math.floor((lo + hi + 1) / 2);
    if (offsets[mid] <= target) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

// ─── Hook ───────────────────────────────────────────────────────────────────

export default function useVariableVirtualList({
  itemCount,
  estimatedItemHeight = 200, // used until real height is measured
  overscan = 3,
}) {
  const containerRef = useRef(null);
  const itemHeightsRef = useRef(new Array(itemCount).fill(estimatedItemHeight));
  const offsetsRef = useRef(buildOffsets(itemHeightsRef.current));
  const observersRef = useRef(new Map()); // index -> ResizeObserver

  const [scrollTop, setScrollTop] = useState(0);
  const [containerHeight, setContainerHeight] = useState(500);

  // Cheap re-render trigger — avoids storing offsets in state to prevent
  // extra allocations on every height measurement.
  const [, forceUpdate] = useReducer((x) => x + 1, 0);

  // Recalculate prefix sums from `fromIndex` onwards (avoids full rebuild).
  const recalcOffsets = useCallback(
    (fromIndex) => {
      const heights = itemHeightsRef.current;
      const offsets = offsetsRef.current;
      const start = fromIndex === 0 ? 0 : offsets[fromIndex]; // anchor
      offsets[fromIndex] = fromIndex === 0 ? 0 : start;
      for (let i = fromIndex; i < itemCount; i++) {
        offsets[i + 1] = offsets[i] + heights[i];
      }
    },
    [itemCount]
  );

  // Re-initialise when itemCount changes (e.g. after a filter change).
  useEffect(() => {
    itemHeightsRef.current = new Array(itemCount).fill(estimatedItemHeight);
    offsetsRef.current = buildOffsets(itemHeightsRef.current);
    // Disconnect all stale observers.
    observersRef.current.forEach((ro) => ro.disconnect());
    observersRef.current.clear();
    forceUpdate();
  }, [itemCount, estimatedItemHeight]);

  // Container scroll + resize listeners.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    setContainerHeight(container.clientHeight);

    const onScroll = () => setScrollTop(container.scrollTop);
    container.addEventListener("scroll", onScroll, { passive: true });

    const ro = new ResizeObserver(([entry]) => {
      setContainerHeight(entry.contentRect.height);
    });
    ro.observe(container);

    return () => {
      container.removeEventListener("scroll", onScroll);
      ro.disconnect();
    };
  }, []);

  // Cleanup all item observers on unmount.
  useEffect(() => {
    return () => observersRef.current.forEach((ro) => ro.disconnect());
  }, []);

  /**
   * Call this with an item's index to get a callback ref.
   * Each rendered item should attach this ref to its outermost DOM node.
   *
   * Example:  <div ref={getMeasureRef(index)} ...>
   */
  const getMeasureRef = useCallback(
    (index) => (node) => {
      // Disconnect any previous observer for this slot.
      if (observersRef.current.has(index)) {
        observersRef.current.get(index).disconnect();
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
    [recalcOffsets]
  );

  // ── Compute visible range ────────────────────────────────────────────────
  const offsets = offsetsRef.current;
  const totalHeight = offsets[itemCount] ?? itemCount * estimatedItemHeight;

  const startIndex = Math.max(
    0,
    binarySearchOffset(offsets, scrollTop) - overscan
  );

  let endIndex = startIndex;
  while (
    endIndex < itemCount - 1 &&
    offsets[endIndex + 1] < scrollTop + containerHeight
  ) {
    endIndex++;
  }
  endIndex = Math.min(itemCount - 1, endIndex + overscan);

  const virtualItems = [];
  for (let i = startIndex; i <= endIndex; i++) {
    virtualItems.push({
      index: i,
      offsetTop: offsets[i],
      size: itemHeightsRef.current[i], // last known height (may be estimated)
    });
  }

  return {
    containerRef,
    virtualItems,
    totalHeight,
    getMeasureRef, // attach to each rendered item's DOM node
    startIndex,
    endIndex,
  };
}
