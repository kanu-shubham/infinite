import { useReducer, useEffect, useCallback, useRef } from "react";
import { fetchHotels } from "../services/hotelService";

const initialState = {
  hotels: [],
  page: 1,
  hasMore: true,
  isLoading: true,
  error: null,
  totalCount: 0,
};

function reducer(state, action) {
  switch (action.type) {
    case "RESET":
      return { ...initialState };
    case "LOADING":
      return { ...state, isLoading: true, error: null };
    case "SUCCESS":
      return {
        ...state,
        hotels:
          action.page === 1
            ? action.data
            : [...state.hotels, ...action.data],
        hasMore: action.hasMore,
        totalCount: action.total,
        isLoading: false,
        error: null,
      };
    case "ERROR":
      return { ...state, isLoading: false, error: action.message };
    case "LOAD_MORE":
      return { ...state, page: state.page + 1 };
    default:
      return state;
  }
}

/**
 * Paginated hotel fetching with infinite scroll.
 * Accepts filters and sortBy from outside (owned by useHotelFilters).
 */
export default function useHotels({ filters, sortBy }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  const requestIdRef = useRef(0);

  // Reset and refetch from page 1 whenever filters or sort changes
  useEffect(() => {
    dispatch({ type: "RESET" });
  }, [filters, sortBy]);

  // Fetch current page
  useEffect(() => {
    const rid = ++requestIdRef.current;

    dispatch({ type: "LOADING" });

    fetchHotels({ filters, sortBy, page: state.page })
      .then((result) => {
        if (rid !== requestIdRef.current) return;
        dispatch({
          type: "SUCCESS",
          data: result.data,
          hasMore: result.hasMore,
          total: result.total,
          page: state.page,
        });
      })
      .catch((err) => {
        if (rid !== requestIdRef.current) return;
        dispatch({ type: "ERROR", message: err.message });
      });
  }, [state.page, filters, sortBy]); // eslint-disable-line

  const loadMore = useCallback(() => {
    if (!state.isLoading && state.hasMore) {
      dispatch({ type: "LOAD_MORE" });
    }
  }, [state.isLoading, state.hasMore]);

  const retry = useCallback(() => {
    dispatch({ type: "LOADING" });
    const rid = ++requestIdRef.current;
    fetchHotels({ filters, sortBy, page: state.page })
      .then((result) => {
        if (rid !== requestIdRef.current) return;
        dispatch({
          type: "SUCCESS",
          data: result.data,
          hasMore: result.hasMore,
          total: result.total,
          page: state.page,
        });
      })
      .catch((err) => {
        if (rid !== requestIdRef.current) return;
        dispatch({ type: "ERROR", message: err.message });
      });
  }, [filters, sortBy, state.page]);

  return {
    hotels: state.hotels,
    hasMore: state.hasMore,
    isLoading: state.isLoading,
    error: state.error,
    totalCount: state.totalCount,
    loadMore,
    retry,
  };
}
