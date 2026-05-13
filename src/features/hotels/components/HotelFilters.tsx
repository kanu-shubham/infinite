import React, { ChangeEvent, useCallback } from 'react';
import { PRICE_RANGES, RATING_OPTIONS } from '../constants';
import type { HotelFilters as HotelFiltersState } from '../types';
import './HotelFilters.css';

export interface HotelFiltersProps {
  filters: HotelFiltersState;
  onFilterChange: <K extends keyof HotelFiltersState>(
    key: K,
    value: HotelFiltersState[K],
  ) => void;
  onReset: () => void;
  hasActiveFilters: boolean;
}

export default function HotelFilters({
  filters,
  onFilterChange,
  onReset,
  hasActiveFilters,
}: HotelFiltersProps): JSX.Element {
  const handleSearchChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => onFilterChange('search', e.target.value),
    [onFilterChange],
  );

  const handlePriceChange = useCallback(
    (e: ChangeEvent<HTMLSelectElement>) => onFilterChange('priceRange', e.target.value),
    [onFilterChange],
  );

  const handleRatingChange = useCallback(
    (e: ChangeEvent<HTMLSelectElement>) => onFilterChange('minRating', e.target.value),
    [onFilterChange],
  );

  return (
    <div className="hotel-filters">
      <div className="hotel-filters__field">
        <label className="hotel-filters__label" htmlFor="search">Search</label>
        <input
          id="search"
          className="hotel-filters__input"
          type="text"
          placeholder="Hotel name or location..."
          value={filters.search}
          onChange={handleSearchChange}
        />
      </div>

      <div className="hotel-filters__field">
        <label className="hotel-filters__label" htmlFor="price">Price Range</label>
        <select
          id="price"
          className="hotel-filters__select"
          value={filters.priceRange}
          onChange={handlePriceChange}
        >
          {PRICE_RANGES.map((opt) => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>
      </div>

      <div className="hotel-filters__field">
        <label className="hotel-filters__label" htmlFor="rating">Minimum Rating</label>
        <select
          id="rating"
          className="hotel-filters__select"
          value={filters.minRating}
          onChange={handleRatingChange}
        >
          {RATING_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>
      </div>

      {hasActiveFilters && (
        <button className="hotel-filters__reset" onClick={onReset}>
          Clear Filters
        </button>
      )}
    </div>
  );
}
