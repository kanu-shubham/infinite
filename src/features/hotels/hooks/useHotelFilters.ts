import { useCallback, useMemo, useState } from 'react';
import useDebounce from '../../../hooks/useDebounce';
import type { HotelFilters, SortKey } from '../types';

const initialRaw: HotelFilters = { search: '', priceRange: '', minRating: '' };

export interface UseHotelFiltersResult {
  /** Bind to controlled inputs — updates instantly. */
  rawFilters: HotelFilters;
  /** Debounced search version — pass to data hooks. */
  filters: HotelFilters;
  sortBy: SortKey | '';
  hasActiveFilters: boolean;
  updateFilter: <K extends keyof HotelFilters>(key: K, value: HotelFilters[K]) => void;
  updateSort: (value: SortKey | '') => void;
  resetFilters: () => void;
}

export default function useHotelFilters(): UseHotelFiltersResult {
  const [rawFilters, setRawFilters] = useState<HotelFilters>(initialRaw);
  const [sortBy, setSortBy] = useState<SortKey | ''>('');

  const debouncedSearch = useDebounce(rawFilters.search, 350);

  const filters = useMemo<HotelFilters>(
    () => ({ ...rawFilters, search: debouncedSearch }),
    [rawFilters, debouncedSearch],
  );

  const updateFilter = useCallback(
    <K extends keyof HotelFilters>(key: K, value: HotelFilters[K]) => {
      setRawFilters((prev) => ({ ...prev, [key]: value }));
    },
    [],
  );

  const updateSort = useCallback((value: SortKey | '') => setSortBy(value), []);

  const resetFilters = useCallback(() => {
    setRawFilters(initialRaw);
    setSortBy('');
  }, []);

  const hasActiveFilters = Boolean(
    rawFilters.search || rawFilters.priceRange || rawFilters.minRating,
  );

  return {
    rawFilters,
    filters,
    sortBy,
    hasActiveFilters,
    updateFilter,
    updateSort,
    resetFilters,
  };
}
