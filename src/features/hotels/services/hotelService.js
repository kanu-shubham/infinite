import { mockHotels } from "../data/mockHotels";
import { HOTELS } from "../constants";

function applyFilters(hotels, filters) {
  let result = hotels;

  if (filters.priceRange) {
    const [min, max] = filters.priceRange.split("-").map(Number);
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
        h.location.toLowerCase().includes(term)
    );
  }

  return result;
}

function applySorting(hotels, sortBy) {
  if (!sortBy) return hotels;

  const [field, direction] = sortBy.split("_");
  const sorted = [...hotels].sort((a, b) => {
    if (direction === "asc") return a[field] - b[field];
    return b[field] - a[field];
  });

  return sorted;
}

function applyPagination(hotels, page, pageSize) {
  const start = 0;
  const end = page * pageSize;
  return {
    data: hotels.slice(start, end),
    total: hotels.length,
    hasMore: end < hotels.length,
  };
}

export function fetchHotels({ filters = {}, sortBy = "", page = 1 } = {}) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (HOTELS.errorRate > 0 && Math.random() < HOTELS.errorRate) {
        reject(new Error("Failed to fetch hotels. Please try again."));
        return;
      }

      try {
        const filtered = applyFilters(mockHotels, filters);
        const sorted = applySorting(filtered, sortBy);
        const paginated = applyPagination(sorted, page, HOTELS.pageSize);

        resolve(paginated);
      } catch (err) {
        reject(new Error("An unexpected error occurred."));
      }
    }, HOTELS.simulatedDelay);
  });
}
