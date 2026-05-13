import { useCallback, useEffect, useRef, useState } from 'react';
import { fetchAllHotels } from '../services/hotelService';
import type { Hotel, HotelFilters, SortKey } from '../types';

export interface UseAllHotelsArgs {
  filters: Partial<HotelFilters>;
  sortBy: SortKey | '';
}

export interface UseAllHotelsResult {
  hotels: Hotel[];
  isLoading: boolean;
  error: string | null;
  retry: () => void;
}

/**
 * Fetches all hotels matching the current filters + sort in one shot.
 * Intended for use with the virtualized list — pagination isn't needed
 * because virtualization handles render cost.
 */
export default function useAllHotels({ filters, sortBy }: UseAllHotelsArgs): UseAllHotelsResult {
  const [hotels, setHotels] = useState<Hotel[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const requestIdRef = useRef(0);

  const load = useCallback(async (currentFilters: Partial<HotelFilters>, currentSort: SortKey | '') => {
    const rid = ++requestIdRef.current;
    setIsLoading(true);
    setError(null);

    try {
      const data = await fetchAllHotels({ filters: currentFilters, sortBy: currentSort });
      if (rid !== requestIdRef.current) return;
      setHotels(data);
    } catch (err) {
      if (rid !== requestIdRef.current) return;
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      if (rid === requestIdRef.current) setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    load(filters, sortBy);
  }, [filters, sortBy, load]);

  const retry = useCallback(() => load(filters, sortBy), [filters, sortBy, load]);

  return { hotels, isLoading, error, retry };
}
