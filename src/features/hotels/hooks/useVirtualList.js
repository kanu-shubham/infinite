import { useState, useEffect, useRef, useCallback, useMemo } from "react";

/**
 * Container-based virtualizer (default strategy).
 *
 * Attaches scroll + resize listeners to a container element.
 * Only renders items within the visible viewport of that container,
 * plus `overscan` buffer items above and below.
 *
 * Usage:
 *   const { containerRef, virtualItems, totalHeight } = useVirtualList({
 *     itemCount: hotels.length,
 *     itemHeight: ITEM_HEIGHT,
 *   });
 *
 *   <div ref={containerRef} style={{ height: 600, overflowY: "auto" }}>
 *     <div style={{ height: totalHeight, position: "relative" }}>
 *       {virtualItems.map(({ index, offsetTop }) => (
 *         <div key={index} style={{ position: "absolute", top: offsetTop, width: "100%" }}>
 *           <MyItem item={data[index]} />
 *         </div>
 *       ))}
 *     </div>
 *   </div>
 */
export default function useVirtualList({
  itemCount,
  itemHeight,
  overscan = 3,
}) {
  const containerRef = useRef(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [containerHeight, setContainerHeight] = useState(500);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // Measure initial height
    setContainerHeight(container.clientHeight);

    const onScroll = () => setScrollTop(container.scrollTop);
    container.addEventListener("scroll", onScroll, { passive: true });

    // Track container resize (e.g. window resize changes panel width)
    const ro = new ResizeObserver(([entry]) => {
      setContainerHeight(entry.contentRect.height);
    });
    ro.observe(container);

    return () => {
      container.removeEventListener("scroll", onScroll);
      ro.disconnect();
    };
  }, []);

  const totalHeight = itemCount * itemHeight;

  const startIndex = Math.max(0, Math.floor(scrollTop / itemHeight) - overscan);
  const endIndex = Math.min(
    itemCount - 1,
    Math.ceil((scrollTop + containerHeight) / itemHeight) - 1 + overscan
  );

  const virtualItems = useMemo(() => {
    const items = [];
    for (let i = startIndex; i <= endIndex; i++) {
      items.push({ index: i, offsetTop: i * itemHeight });
    }
    return items;
  }, [startIndex, endIndex, itemHeight]);

  return { containerRef, virtualItems, totalHeight, startIndex, endIndex };
}
