import { useState, useEffect, useRef, useMemo } from "react";

/**
 * Window-based virtualizer (alternative strategy).
 *
 * Uses the browser window as the scroll container.
 * Tracks window.scrollY and accounts for the list's offsetTop so items
 * that are above the fold (but below the list start) are excluded.
 *
 * Usage:
 *   const { listRef, virtualItems, totalHeight } = useWindowVirtualizer({
 *     itemCount: hotels.length,
 *     itemHeight: ITEM_HEIGHT,
 *   });
 *
 *   <div ref={listRef} style={{ position: "relative", height: totalHeight }}>
 *     {virtualItems.map(({ index, offsetTop }) => (
 *       <div key={index} style={{ position: "absolute", top: offsetTop, width: "100%" }}>
 *         <MyItem item={data[index]} />
 *       </div>
 *     ))}
 *   </div>
 */
export default function useWindowVirtualizer({
  itemCount,
  itemHeight,
  overscan = 3,
}) {
  const listRef = useRef(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [listOffsetTop, setListOffsetTop] = useState(0);
  const [viewportHeight, setViewportHeight] = useState(
    typeof window !== "undefined" ? window.innerHeight : 800
  );

  // Measure list's distance from document top once mounted (and on resize)
  useEffect(() => {
    const measureOffset = () => {
      if (listRef.current) {
        setListOffsetTop(
          listRef.current.getBoundingClientRect().top + window.scrollY
        );
      }
    };
    measureOffset();
    window.addEventListener("resize", measureOffset);
    return () => window.removeEventListener("resize", measureOffset);
  }, []);

  useEffect(() => {
    const onScroll = () => setScrollTop(window.scrollY);
    const onResize = () => setViewportHeight(window.innerHeight);

    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onResize);

    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onResize);
    };
  }, []);

  const totalHeight = itemCount * itemHeight;

  // Scroll position relative to the list's own top edge
  const scrollRelative = Math.max(0, scrollTop - listOffsetTop);

  const startIndex = Math.max(
    0,
    Math.floor(scrollRelative / itemHeight) - overscan
  );
  const endIndex = Math.min(
    itemCount - 1,
    Math.ceil((scrollRelative + viewportHeight) / itemHeight) - 1 + overscan
  );

  const virtualItems = useMemo(() => {
    const items = [];
    for (let i = startIndex; i <= endIndex; i++) {
      items.push({ index: i, offsetTop: i * itemHeight });
    }
    return items;
  }, [startIndex, endIndex, itemHeight]);

  return { listRef, virtualItems, totalHeight, startIndex, endIndex };
}
