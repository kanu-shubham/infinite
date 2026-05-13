import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import HotelFilters from './HotelFilters';
import type { HotelFilters as F } from '../types';

const empty: F = { search: '', priceRange: '', minRating: '' };

describe('<HotelFilters /> a11y + behaviour', () => {
  test('renders inside a search landmark with a label', () => {
    render(
      <HotelFilters
        filters={empty}
        onFilterChange={() => {}}
        onReset={() => {}}
        hasActiveFilters={false}
      />,
    );
    expect(screen.getByRole('search', { name: /filter hotels/i })).toBeInTheDocument();
  });

  test('inputs are correctly labelled', () => {
    render(
      <HotelFilters
        filters={empty}
        onFilterChange={() => {}}
        onReset={() => {}}
        hasActiveFilters={false}
      />,
    );
    expect(screen.getByLabelText(/search/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/price range/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/minimum rating/i)).toBeInTheDocument();
  });

  test('typing in search emits onFilterChange("search", value)', () => {
    const onChange = jest.fn();
    render(
      <HotelFilters
        filters={empty}
        onFilterChange={onChange}
        onReset={() => {}}
        hasActiveFilters={false}
      />,
    );
    fireEvent.change(screen.getByLabelText(/search/i), { target: { value: 'paris' } });
    expect(onChange).toHaveBeenCalledWith('search', 'paris');
  });

  test('hides Clear Filters when no filters are active', () => {
    render(
      <HotelFilters
        filters={empty}
        onFilterChange={() => {}}
        onReset={() => {}}
        hasActiveFilters={false}
      />,
    );
    expect(screen.queryByRole('button', { name: /clear all active filters/i })).toBeNull();
  });

  test('shows Clear Filters with descriptive aria-label and fires onReset', () => {
    const onReset = jest.fn();
    render(
      <HotelFilters
        filters={{ ...empty, search: 'x' }}
        onFilterChange={() => {}}
        onReset={onReset}
        hasActiveFilters
      />,
    );
    const btn = screen.getByRole('button', { name: /clear all active filters/i });
    fireEvent.click(btn);
    expect(onReset).toHaveBeenCalledTimes(1);
  });
});
