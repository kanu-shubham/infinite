/**
 * useClientSideHotels — fetch once, filter/sort/paginate in browser
 * ─────────────────────────────────────────────────────────────────────────────
 * HOW PAGINATION CONNECTS TO THE LIST
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *  rawHotels[] (full dataset, fetched once, lives in memory the whole time)
 *      │
 *      ▼  useMemo — applyFilters(rawHotels, filters)
 *  filteredHotels[]
 *      │
 *      ▼  useMemo — applySort(filteredHotels, sortBy)
 *  sortedHotels[]            ← totalCount = sortedHotels.length
 *      │
 *      ▼  useMemo — applyPage(sortedHotels, page, mode)
 *  visibleHotels[]           ← what the component renders
 *
 *
 *  MODE A — Infinite scroll  (isPaginated = false, default)
 *  ──────────────────────────────────────────────────────────
 *
 *  page=1 → slice(0,  8) → [item0 ..item7]
 *  page=2 → slice(0, 16) → [item0 ..item15]   ← GROWING window
 *  page=3 → slice(0, 24) → [item0 ..item23]
 *
 *  All previously seen cards stay mounted (DOM grows).
 *  No network request — just expanding the slice ceiling.
 *
 *
 *  MODE B — Traditional pagination  (isPaginated = true)
 *  ──────────────────────────────────────────────────────
 *
 *  page=1 → slice(0,  8) → [item0 ..item7]
 *  page=2 → slice(8, 16) → [item8 ..item15]   ← SLIDING window
 *  page=3 → slice(16,24) → [item16..item23]
 *
 *  Only the current page's cards are mounted (constant DOM size).
 *  No network request — just moving the slice window.
 *
 *
 *  The only difference between the two modes:
 *    Infinite:   slice(0,          page * PAGE_SIZE)
 *    Paginated:  slice((page-1) * PAGE_SIZE, page * PAGE_SIZE)
 */

import { useState, useEffect, useMemo, useCallback } from "react";
import { fetchAllHotels } from "../services/hotelService";
import { HOTELS } from "../constants";

const { pageSize: PAGE_SIZE } = HOTELS;

export default function useClientSideHotels({ filters, sortBy, isPaginated = false }) {
  // ── 1. Source of truth — fetched once ───────────────────────────────────
  const [rawHotels, setRawHotels] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError]         = useState(null);

  // ── 2. Only real state — everything else is derived ─────────────────────
  const [page, setPage] = useState(1);

  // ── 3. Fetch once on mount ───────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    fetchAllHotels()
      .then((data) => { if (!cancelled) { setRawHotels(data); setIsLoading(false); } })
      .catch((err) => { if (!cancelled) { setError(err.message); setIsLoading(false); } });
    return () => { cancelled = true; };
  }, []); // intentionally empty — one fetch, ever

  // ── 4. Reset page when filters/sort change ───────────────────────────────
  useEffect(() => { setPage(1); }, [filters, sortBy]);

  // ── 5. Filter — pure derivation ──────────────────────────────────────────
  const filteredHotels = useMemo(() => {
    let result = rawHotels;
    if (filters.priceRange) {
      const [min, max] = filters.priceRange.split("-").map(Number);
      result = result.filter((h) => h.price >= min && h.price <= max);
    }
    if (filters.minRating) {
      result = result.filter((h) => h.rating >= parseFloat(filters.minRating));
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

  // ── 6. Sort — derives from filteredHotels ────────────────────────────────
  const sortedHotels = useMemo(() => {
    if (!sortBy) return filteredHotels;
    const [field, direction] = sortBy.split("_");
    return [...filteredHotels].sort((a, b) =>
      direction === "asc" ? a[field] - b[field] : b[field] - a[field]
    );
  }, [filteredHotels, sortBy]);

  // ── 7. Paginate — the ONLY difference between modes ─────────────────────
  const visibleHotels = useMemo(() => {
    if (isPaginated) {
      // Sliding window — only this page's items are in the array
      const start = (page - 1) * PAGE_SIZE;
      return sortedHotels.slice(start, start + PAGE_SIZE);
    }
    // Growing window — all items up to current page ceiling
    return sortedHotels.slice(0, page * PAGE_SIZE);
  }, [sortedHotels, page, isPaginated]);

  const totalCount = sortedHotels.length;
  const totalPages = Math.ceil(totalCount / PAGE_SIZE) || 1;

  // ── Infinite scroll API ──────────────────────────────────────────────────
  const hasMore = !isPaginated && visibleHotels.length < sortedHotels.length;

  const loadMore = useCallback(() => {
    if (hasMore) setPage((p) => p + 1);
  }, [hasMore]);

  // ── Traditional pagination API ───────────────────────────────────────────
  const goToPage = useCallback(
    (newPage) => {
      if (newPage >= 1 && newPage <= totalPages) setPage(newPage);
    },
    [totalPages]
  );

  const retry = useCallback(() => {
    setError(null);
    setIsLoading(true);
    fetchAllHotels()
      .then((data) => { setRawHotels(data); setIsLoading(false); })
      .catch((err) => { setError(err.message); setIsLoading(false); });
  }, []);

  return {
    hotels:      visibleHotels,
    isLoading,
    error,
    totalCount,
    // Infinite scroll
    hasMore,
    loadMore,
    // Traditional pagination
    currentPage: page,
    totalPages,
    goToPage,
    // Shared
    retry,
  };
}
