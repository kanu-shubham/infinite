export interface Hotel {
  id: number;
  name: string;
  location: string;
  price: number;
  rating: number;
  reviewCount: number;
  image: string;
  amenities: string[];
  description: string;
}

export interface HotelFilters {
  search: string;
  priceRange: string;
  minRating: string;
}

export type SortField = 'price' | 'rating' | 'reviewCount';
export type SortDirection = 'asc' | 'desc';
export type SortKey = `${SortField}_${SortDirection}` | '';

export interface PaginatedHotels {
  data: Hotel[];
  total: number;
  hasMore: boolean;
}
