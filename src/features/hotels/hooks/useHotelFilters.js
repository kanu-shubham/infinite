import { useState, useCallback, useMemo } from "react";
import useDebounce from "../../../hooks/useDebounce";

const initialRaw = { search: "", priceRange: "", minRating: "" };

/**
 * Owns all filter + sort state.
 * Exposes `rawFilters` for controlled inputs (instant),
 * and `filters` (debounced search) for passing to data hooks.
 */
export default function useHotelFilters() {
  const [rawFilters, setRawFilters] = useState(initialRaw);
  const [sortBy, setSortBy] = useState("");

  const debouncedSearch = useDebounce(rawFilters.search, 350);

  // Only the search field is debounced — price/rating are instant selects
  const filters = useMemo(
    () => ({ ...rawFilters, search: debouncedSearch }),
    [rawFilters, debouncedSearch]
  );

  const updateFilter = useCallback((key, value) => {
    setRawFilters((prev) => ({ ...prev, [key]: value }));
  }, []);

  const updateSort = useCallback((value) => setSortBy(value), []);

  const resetFilters = useCallback(() => {
    setRawFilters(initialRaw);
    setSortBy("");
  }, []);

  const hasActiveFilters = useMemo(
    () => Boolean(rawFilters.search || rawFilters.priceRange || rawFilters.minRating),
    [rawFilters.search, rawFilters.priceRange, rawFilters.minRating]
  );

  return {
    rawFilters,   // bind to controlled inputs
    filters,      // debounced — pass to data hooks
    sortBy,
    hasActiveFilters,
    updateFilter,
    updateSort,
    resetFilters,
  };
}
