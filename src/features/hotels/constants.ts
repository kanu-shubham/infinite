import type { SortKey } from './types';

export const HOTELS = {
  pageSize: 8,
  simulatedDelay: 600,
  errorRate: 0,
} as const;

export interface SortOption {
  value: SortKey;
  label: string;
}

export const SORT_OPTIONS: ReadonlyArray<SortOption> = [
  { value: 'price_asc',         label: 'Price: Low to High' },
  { value: 'price_desc',        label: 'Price: High to Low' },
  { value: 'rating_desc',       label: 'Rating: High to Low' },
  { value: 'rating_asc',        label: 'Rating: Low to High' },
  { value: 'reviewCount_desc',  label: 'Most Reviewed' },
];

export interface RangeOption {
  value: string;
  label: string;
}

export const PRICE_RANGES: ReadonlyArray<RangeOption> = [
  { value: '',        label: 'Any Price' },
  { value: '0-100',   label: 'Under $100' },
  { value: '100-200', label: '$100 - $200' },
  { value: '200-350', label: '$200 - $350' },
  { value: '350-999', label: '$350+' },
];

export const RATING_OPTIONS: ReadonlyArray<RangeOption> = [
  { value: '',    label: 'Any Rating' },
  { value: '4.5', label: '4.5+ Excellent' },
  { value: '4',   label: '4+ Very Good' },
  { value: '3.5', label: '3.5+ Good' },
  { value: '3',   label: '3+ Fair' },
];
