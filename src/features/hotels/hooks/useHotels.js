import { useState, useCallback, useEffect, useRef } from "react";
import { fetchHotels } from "../services/hotelService";

const initialFilters = {
  priceRange: "",
  minRating: "",
  search: "",
};

export default function useHotels() {
  const [hotels, setHotels] = useState([]);
  const [filters, setFilters] = useState(initialFilters);
  const [sortBy, setSortBy] = useState("");
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(true);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [totalCount, setTotalCount] = useState(0);

  const requestIdRef = useRef(0);

  const loadHotels = useCallback(async (currentFilters, currentSort, currentPage) => {
    const requestId = ++requestIdRef.current;
    setIsLoading(true);
    setError(null);

    try {
      const result = await fetchHotels({
        filters: currentFilters,
        sortBy: currentSort,
        page: currentPage,
      });

      if (requestId !== requestIdRef.current) return;

      setHotels(result.data);
      setHasMore(result.hasMore);
      setTotalCount(result.total);
    } catch (err) {
      if (requestId !== requestIdRef.current) return;
      setError(err.message);
    } finally {
      if (requestId === requestIdRef.current) {
        setIsLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    loadHotels(filters, sortBy, page);
  }, [filters, sortBy, page, loadHotels]);

  const updateFilter = useCallback((key, value) => {
    setFilters((prev) => ({ ...prev, [key]: value }));
    setPage(1);
  }, []);

  const updateSort = useCallback((value) => {
    setSortBy(value);
    setPage(1);
  }, []);

  const loadMore = useCallback(() => {
    if (!isLoading && hasMore) {
      setPage((prev) => prev + 1);
    }
  }, [isLoading, hasMore]);

  const retry = useCallback(() => {
    loadHotels(filters, sortBy, page);
  }, [filters, sortBy, page, loadHotels]);

  const resetFilters = useCallback(() => {
    setFilters(initialFilters);
    setSortBy("");
    setPage(1);
  }, []);

  return {
    hotels,
    filters,
    sortBy,
    hasMore,
    isLoading,
    error,
    totalCount,
    updateFilter,
    updateSort,
    loadMore,
    retry,
    resetFilters,
  };
}
