/**
 * useClientSideHotels
 * ─────────────────────────────────────────────────────────────────────────────
 * ARCHITECTURE: Fetch all data ONCE on mount. All filtering, sorting, and
 * pagination are synchronous useMemo computations — no re-fetching on filter
 * change, no loading state between filter changes.
 *
 * When to use this vs the server-side approach (useHotels):
 *
 *  Client-side (this hook)        │  Server-side (useHotels)
 * ─────────────────────────────────┼────────────────────────────────────────
 *  Small–medium dataset (<10k)     │  Large dataset (10k+ items)
 *  Filters feel instant (0ms)      │  Each filter fires a network request
 *  One network request total       │  N requests (one per filter/page change)
 *  Full dataset in browser memory  │  Only current page in memory
 *  No skeleton between filters     │  Can show skeleton/spinner per filter
 *  Simple state (useMemo only)     │  Complex async state (useReducer)
 *
 * KEY INSIGHT:
 *   rawHotels (source of truth, never modified)
 *       │
 *       ▼  useMemo — filter
 *   filteredHotels
 *       │
 *       ▼  useMemo — sort
 *   sortedHotels
 *       │
 *       ▼  useMemo — paginate (slice)
 *   visibleHotels  ← what the component renders
 *
 *   Each layer only recomputes when its inputs change.
 */

import { useState, useEffect, useMemo, useCallback } from "react";
import { fetchAllHotels } from "../services/hotelService";

const PAGE_SIZE = 8;

export default function useClientSideHotels({ filters, sortBy }) {
  // ── 1. Source of truth — fetched once, never changes ────────────────────
  const [rawHotels, setRawHotels]   = useState([]);
  const [isLoading, setIsLoading]   = useState(true);  // only true on initial fetch
  const [error, setError]           = useState(null);

  // ── 2. Pagination — the only piece of state that isn't derived ───────────
  const [page, setPage] = useState(1);

  // ── 3. Fetch once on mount — empty dep array is intentional ─────────────
  useEffect(() => {
    let cancelled = false;

    fetchAllHotels()               // no filters — get everything
      .then((data) => {
        if (!cancelled) {
          setRawHotels(data);
          setIsLoading(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err.message);
          setIsLoading(false);
        }
      });

    return () => { cancelled = true; };
  }, []); // ← intentionally empty — one fetch, ever

  // ── 4. Reset to page 1 when filters or sort change ──────────────────────
  //   No async involved — the useMemo below runs synchronously on the same
  //   render that processes the filter change. setPage(1) triggers one extra
  //   render to trim the paginated slice, which is negligible.
  useEffect(() => {
    setPage(1);
  }, [filters, sortBy]);

  // ── 5. Filtering — pure synchronous derivation ──────────────────────────
  //   Only recomputes when rawHotels or filters change.
  //   No network request. No loading state. Runs in <1ms for <10k items.
  const filteredHotels = useMemo(() => {
    let result = rawHotels;

    if (filters.priceRange) {
      const [min, max] = filters.priceRange.split("-").map(Number);
      result = result.filter((h) => h.price >= min && h.price <= max);
    }

    if (filters.minRating) {
      const min = parseFloat(filters.minRating);
      result = result.filter((h) => h.rating >= min);
    }

    if (filters.search) {
      const term = filters.search.toLowerCase();
      result = result.filter(
        (h) =>
          h.name.toLowerCase().includes(term) ||
          h.location.toLowerCase().includes(term)
      );
    }

    return result;
  }, [rawHotels, filters]);

  // ── 6. Sorting — derives from filteredHotels ────────────────────────────
  //   Separated from filtering so that a sort change doesn't re-run the
  //   filter logic and vice versa.
  const sortedHotels = useMemo(() => {
    if (!sortBy) return filteredHotels;

    const [field, direction] = sortBy.split("_");
    return [...filteredHotels].sort((a, b) =>
      direction === "asc" ? a[field] - b[field] : b[field] - a[field]
    );
  }, [filteredHotels, sortBy]);

  // ── 7. Pagination — a simple slice of sorted results ────────────────────
  //   "Infinite scroll" here just means increasing the slice ceiling.
  //   The full dataset is already in memory; we just expose more of it.
  const visibleHotels = useMemo(
    () => sortedHotels.slice(0, page * PAGE_SIZE),
    [sortedHotels, page]
  );

  const hasMore      = visibleHotels.length < sortedHotels.length;
  const totalCount   = sortedHotels.length; // post-filter count

  const loadMore = useCallback(() => {
    // No async, no loading state — just increment the slice ceiling.
    if (hasMore) setPage((p) => p + 1);
  }, [hasMore]);

  const retry = useCallback(() => {
    setError(null);
    setIsLoading(true);
    fetchAllHotels()
      .then((data) => { setRawHotels(data); setIsLoading(false); })
      .catch((err) => { setError(err.message); setIsLoading(false); });
  }, []);

  return {
    hotels: visibleHotels,
    hasMore,
    isLoading,   // true ONLY during the initial fetch — never during filter changes
    error,
    totalCount,
    loadMore,
    retry,
  };
}
