import React, { memo, useCallback } from "react";
import { SORT_OPTIONS } from "../constants";
import "./HotelSort.css";

const HotelSort = memo(function HotelSort({ value, onChange, totalCount }) {
  const handleChange = useCallback(
    (e) => onChange(e.target.value),
    [onChange]
  );

  return (
    <div className="hotel-sort">
      <span className="hotel-sort__count">
        {totalCount} {totalCount === 1 ? "hotel" : "hotels"} found
      </span>
      <div className="hotel-sort__control">
        <label className="hotel-sort__label" htmlFor="sortBy">
          Sort by:
        </label>
        <select
          id="sortBy"
          className="hotel-sort__select"
          value={value}
          onChange={handleChange}
        >
          <option value="">Default</option>
          {SORT_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
});

export default HotelSort;
