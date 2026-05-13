import React, { ChangeEvent, useCallback } from 'react';
import { SORT_OPTIONS } from '../constants';
import type { SortKey } from '../types';
import './HotelSort.css';

export interface HotelSortProps {
  value: SortKey | '';
  onChange: (value: SortKey | '') => void;
  totalCount: number;
}

export default function HotelSort({ value, onChange, totalCount }: HotelSortProps): JSX.Element {
  const handleChange = useCallback(
    (e: ChangeEvent<HTMLSelectElement>) => onChange(e.target.value as SortKey | ''),
    [onChange],
  );

  return (
    <div className="hotel-sort">
      <span className="hotel-sort__count">
        {totalCount} {totalCount === 1 ? 'hotel' : 'hotels'} found
      </span>
      <div className="hotel-sort__control">
        <label className="hotel-sort__label" htmlFor="sortBy">Sort by:</label>
        <select
          id="sortBy"
          className="hotel-sort__select"
          value={value}
          onChange={handleChange}
        >
          <option value="">Default</option>
          {SORT_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>
      </div>
    </div>
  );
}
