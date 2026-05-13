import { useCallback, useEffect, useReducer, useRef } from 'react';
import { fetchHotels } from '../services/hotelService';
import type { Hotel, HotelFilters, PaginatedHotels, SortKey } from '../types';

interface State {
  hotels: Hotel[];
  page: number;
  hasMore: boolean;
  isLoading: boolean;
  error: string | null;
  totalCount: number;
}

type Action =
  | { type: 'RESET' }
  | { type: 'LOADING' }
  | { type: 'SUCCESS'; data: Hotel[]; hasMore: boolean; total: number; page: number }
  | { type: 'ERROR'; message: string }
  | { type: 'LOAD_MORE' };

const initialState: State = {
  hotels: [],
  page: 1,
  hasMore: true,
  isLoading: true,
  error: null,
  totalCount: 0,
};

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case 'RESET':
      return { ...initialState };
    case 'LOADING':
      return { ...state, isLoading: true, error: null };
    case 'SUCCESS':
      return {
        ...state,
        hotels: action.page === 1 ? action.data : [...state.hotels, ...action.data],
        hasMore: action.hasMore,
        totalCount: action.total,
        isLoading: false,
        error: null,
      };
    case 'ERROR':
      return { ...state, isLoading: false, error: action.message };
    case 'LOAD_MORE':
      return { ...state, page: state.page + 1 };
    default:
      return state;
  }
}

export interface UseHotelsArgs {
  filters: Partial<HotelFilters>;
  sortBy: SortKey | '';
}

export interface UseHotelsResult {
  hotels: Hotel[];
  hasMore: boolean;
  isLoading: boolean;
  error: string | null;
  totalCount: number;
  loadMore: () => void;
  retry: () => void;
}

export default function useHotels({ filters, sortBy }: UseHotelsArgs): UseHotelsResult {
  const [state, dispatch] = useReducer(reducer, initialState);
  const requestIdRef = useRef(0);

  useEffect(() => {
    dispatch({ type: 'RESET' });
  }, [filters, sortBy]);

  useEffect(() => {
    const rid = ++requestIdRef.current;
    dispatch({ type: 'LOADING' });

    fetchHotels({ filters, sortBy, page: state.page })
      .then((result: PaginatedHotels) => {
        if (rid !== requestIdRef.current) return;
        dispatch({
          type: 'SUCCESS',
          data: result.data,
          hasMore: result.hasMore,
          total: result.total,
          page: state.page,
        });
      })
      .catch((err: unknown) => {
        if (rid !== requestIdRef.current) return;
        dispatch({
          type: 'ERROR',
          message: err instanceof Error ? err.message : 'Unknown error',
        });
      });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.page, filters, sortBy]);

  const loadMore = useCallback(() => {
    if (!state.isLoading && state.hasMore) {
      dispatch({ type: 'LOAD_MORE' });
    }
  }, [state.isLoading, state.hasMore]);

  const retry = useCallback(() => {
    dispatch({ type: 'LOADING' });
    const rid = ++requestIdRef.current;
    fetchHotels({ filters, sortBy, page: state.page })
      .then((result: PaginatedHotels) => {
        if (rid !== requestIdRef.current) return;
        dispatch({
          type: 'SUCCESS',
          data: result.data,
          hasMore: result.hasMore,
          total: result.total,
          page: state.page,
        });
      })
      .catch((err: unknown) => {
        if (rid !== requestIdRef.current) return;
        dispatch({
          type: 'ERROR',
          message: err instanceof Error ? err.message : 'Unknown error',
        });
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
