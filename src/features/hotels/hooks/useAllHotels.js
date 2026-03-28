import { useState, useEffect, useCallback, useRef } from "react";
import { fetchAllHotels } from "../services/hotelService";

/**
 * Fetches all hotels matching the current filters + sort in one shot.
 * Intended for use with the virtualized list — no pagination needed
 * because virtualization handles the rendering cost.
 */
export default function useAllHotels({ filters, sortBy }) {
  const [hotels, setHotels] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const requestIdRef = useRef(0);

  const load = useCallback(async (currentFilters, currentSort) => {
    const rid = ++requestIdRef.current;
    setIsLoading(true);
    setError(null);

    try {
      const data = await fetchAllHotels({
        filters: currentFilters,
        sortBy: currentSort,
      });
      if (rid !== requestIdRef.current) return;
      setHotels(data);
    } catch (err) {
      if (rid !== requestIdRef.current) return;
      setError(err.message);
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
