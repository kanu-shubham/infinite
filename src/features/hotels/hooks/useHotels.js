/**
 * useHotels — server-side paginated fetching
 * ─────────────────────────────────────────────────────────────────────────────
 * HOW PAGINATION CONNECTS TO THE LIST
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *  MODE A — Infinite scroll  (isPaginated = false, default)
 *  ──────────────────────────────────────────────────────────
 *
 *   page state ──► fetchHotels(page=1)  ──► result.data [0..7]
 *                                               │
 *                                        hotels = result.data   (replace, page 1)
 *
 *   IntersectionObserver fires loadMore()
 *        │
 *        ▼
 *   page state ──► fetchHotels(page=2)  ──► result.data [0..7]
 *                                               │
 *                                        hotels = [...prev, ...result.data]  (APPEND)
 *
 *   DOM: every fetched hotel card stays mounted and visible.
 *   Memory: grows with each page load.
 *
 *
 *  MODE B — Traditional pagination  (isPaginated = true)
 *  ──────────────────────────────────────────────────────
 *
 *   page state ──► fetchHotels(page=1)  ──► result.data [0..7]
 *                                               │
 *                                        hotels = result.data   (replace)
 *
 *   User clicks page 3 → goToPage(3)
 *        │
 *        ▼
 *   page state ──► fetchHotels(page=3)  ──► result.data [0..7]
 *                                               │
 *                                        hotels = result.data   (REPLACE — not append)
 *
 *   DOM: only current page's cards are mounted.
 *   Memory: constant — previous page cards are unmounted.
 *
 *
 *  The only code difference between the two modes is one line in the reducer:
 *
 *    Infinite:    hotels = page === 1 ? data : [...prev, ...data]
 *    Paginated:   hotels = data   (always replace)
 */

import { useReducer, useEffect, useCallback, useRef } from "react";
import { fetchHotels } from "../services/hotelService";
import { HOTELS } from "../constants";

const initialState = {
  hotels:     [],
  page:       1,
  hasMore:    true,
  isLoading:  true,
  error:      null,
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
        // ── THE KEY DIFFERENCE BETWEEN THE TWO MODES ──────────────────────
        // isPaginated → always replace (user jumped to an exact page)
        // infinite    → append unless we just reset (page 1 after filter change)
        hotels: action.replace
          ? action.data
          : action.page === 1
            ? action.data
            : [...state.hotels, ...action.data],
        hasMore:    action.hasMore,
        totalCount: action.total,
        isLoading:  false,
        error:      null,
      };

    case "ERROR":
      return { ...state, isLoading: false, error: action.message };

    // Infinite scroll: increment page (appends on next fetch)
    case "LOAD_MORE":
      return { ...state, page: state.page + 1 };

    // Traditional pagination: jump to exact page (replaces on next fetch)
    case "SET_PAGE":
      return { ...state, page: action.page };

    default:
      return state;
  }
}

/**
 * @param {object} filters   - from useHotelFilters (debounced)
 * @param {string} sortBy    - from useHotelFilters
 * @param {boolean} isPaginated
 *   false → infinite scroll, exposes loadMore + hasMore
 *   true  → page buttons,   exposes goToPage + currentPage + totalPages
 */
export default function useHotels({ filters, sortBy, isPaginated = false }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  const requestIdRef      = useRef(0);

  // Reset to page 1 whenever filters or sort change
  useEffect(() => {
    dispatch({ type: "RESET" });
  }, [filters, sortBy]);

  // Fetch whenever page, filters, or sort changes
  useEffect(() => {
    const rid = ++requestIdRef.current;
    dispatch({ type: "LOADING" });

    fetchHotels({ filters, sortBy, page: state.page })
      .then((result) => {
        if (rid !== requestIdRef.current) return;
        dispatch({
          type:    "SUCCESS",
          data:    result.data,
          hasMore: result.hasMore,
          total:   result.total,
          page:    state.page,
          replace: isPaginated,   // ← tells reducer to replace, not append
        });
      })
      .catch((err) => {
        if (rid !== requestIdRef.current) return;
        dispatch({ type: "ERROR", message: err.message });
      });
  }, [state.page, filters, sortBy, isPaginated]); // eslint-disable-line

  // ── Infinite scroll API ──────────────────────────────────────────────────
  const loadMore = useCallback(() => {
    if (!state.isLoading && state.hasMore && !isPaginated) {
      dispatch({ type: "LOAD_MORE" });
    }
  }, [state.isLoading, state.hasMore, isPaginated]);

  // ── Traditional pagination API ───────────────────────────────────────────
  const totalPages = Math.ceil(state.totalCount / HOTELS.pageSize) || 1;

  const goToPage = useCallback(
    (newPage) => {
      if (newPage >= 1 && newPage <= totalPages && newPage !== state.page) {
        dispatch({ type: "SET_PAGE", page: newPage });
      }
    },
    [state.page, totalPages]
  );

  const retry = useCallback(() => {
    const rid = ++requestIdRef.current;
    dispatch({ type: "LOADING" });
    fetchHotels({ filters, sortBy, page: state.page })
      .then((result) => {
        if (rid !== requestIdRef.current) return;
        dispatch({
          type:    "SUCCESS",
          data:    result.data,
          hasMore: result.hasMore,
          total:   result.total,
          page:    state.page,
          replace: isPaginated,
        });
      })
      .catch((err) => {
        if (rid !== requestIdRef.current) return;
        dispatch({ type: "ERROR", message: err.message });
      });
  }, [filters, sortBy, state.page, isPaginated]);

  return {
    hotels:      state.hotels,
    isLoading:   state.isLoading,
    error:       state.error,
    totalCount:  state.totalCount,
    // Infinite scroll
    hasMore:     state.hasMore,
    loadMore,
    // Traditional pagination
    currentPage: state.page,
    totalPages,
    goToPage,
    // Shared
    retry,
  };
}
