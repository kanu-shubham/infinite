import { mockHotels } from '../data/mockHotels';
import { HOTELS } from '../constants';
import type { Hotel, HotelFilters, PaginatedHotels, SortField, SortKey } from '../types';

function applyFilters(hotels: Hotel[], filters: Partial<HotelFilters>): Hotel[] {
  let result = hotels;

  if (filters.priceRange) {
    const [min, max] = filters.priceRange.split('-').map(Number);
    result = result.filter((h) => h.price >= min && h.price <= max);
  }

  if (filters.minRating) {
    const minRating = parseFloat(filters.minRating);
    result = result.filter((h) => h.rating >= minRating);
  }

  if (filters.search) {
    const term = filters.search.toLowerCase();
    result = result.filter(
      (h) =>
        h.name.toLowerCase().includes(term) ||
        h.location.toLowerCase().includes(term),
    );
  }

  return result;
}

function applySorting(hotels: Hotel[], sortBy: SortKey | ''): Hotel[] {
  if (!sortBy) return hotels;

  const [field, direction] = sortBy.split('_') as [SortField, 'asc' | 'desc'];
  return [...hotels].sort((a, b) => {
    if (direction === 'asc') return a[field] - b[field];
    return b[field] - a[field];
  });
}

function applyPagination(hotels: Hotel[], page: number, pageSize: number): PaginatedHotels {
  // Return only this page's slice. The infinite-scroll reducer appends
  // `action.data` to existing hotels, so returning the cumulative slice
  // (start=0) would re-include earlier pages and produce duplicates.
  const start = (page - 1) * pageSize;
  const end = page * pageSize;
  return {
    data: hotels.slice(start, end),
    total: hotels.length,
    hasMore: end < hotels.length,
  };
}

export interface FetchOptions {
  filters?: Partial<HotelFilters>;
  sortBy?: SortKey | '';
}

export interface FetchPageOptions extends FetchOptions {
  page?: number;
}

/** Fetches ALL hotels matching filters+sort — used by the virtualized list. */
export function fetchAllHotels({ filters = {}, sortBy = '' }: FetchOptions = {}): Promise<Hotel[]> {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      try {
        const filtered = applyFilters(mockHotels, filters);
        const sorted = applySorting(filtered, sortBy);
        resolve(sorted);
      } catch {
        reject(new Error('An unexpected error occurred.'));
      }
    }, HOTELS.simulatedDelay);
  });
}

/** Fetches a paginated slice — used by the standard infinite-scroll list. */
export function fetchHotels({
  filters = {},
  sortBy = '',
  page = 1,
}: FetchPageOptions = {}): Promise<PaginatedHotels> {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (HOTELS.errorRate > 0 && Math.random() < HOTELS.errorRate) {
        reject(new Error('Failed to fetch hotels. Please try again.'));
        return;
      }

      try {
        const filtered = applyFilters(mockHotels, filters);
        const sorted = applySorting(filtered, sortBy);
        const paginated = applyPagination(sorted, page, HOTELS.pageSize);
        resolve(paginated);
      } catch {
        reject(new Error('An unexpected error occurred.'));
      }
    }, HOTELS.simulatedDelay);
  });
}
