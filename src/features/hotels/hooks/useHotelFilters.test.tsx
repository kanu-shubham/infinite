import React from 'react';
import { act, render } from '@testing-library/react';
import useHotelFilters from './useHotelFilters';

beforeEach(() => jest.useFakeTimers());
afterEach(() => jest.useRealTimers());

type Captured = ReturnType<typeof useHotelFilters>;

function Probe({ onState }: { onState: (s: Captured) => void }): null {
  const state = useHotelFilters();
  onState(state);
  return null;
}

describe('useHotelFilters', () => {
  test('starts with empty filters and sort', () => {
    const captured: Captured[] = [];
    render(<Probe onState={(s) => captured.push(s)} />);
    const s = captured[captured.length - 1];
    expect(s.rawFilters).toEqual({ search: '', priceRange: '', minRating: '' });
    expect(s.sortBy).toBe('');
    expect(s.hasActiveFilters).toBe(false);
  });

  test('updateFilter updates rawFilters instantly and debounces search', () => {
    const captured: Captured[] = [];
    render(<Probe onState={(s) => captured.push(s)} />);

    act(() => captured[captured.length - 1].updateFilter('search', 'hotel'));

    const afterUpdate = captured[captured.length - 1];
    expect(afterUpdate.rawFilters.search).toBe('hotel');
    expect(afterUpdate.filters.search).toBe(''); // not debounced yet
    expect(afterUpdate.hasActiveFilters).toBe(true);

    act(() => { jest.advanceTimersByTime(350); });
    expect(captured[captured.length - 1].filters.search).toBe('hotel');
  });

  test('priceRange / minRating are not debounced', () => {
    const captured: Captured[] = [];
    render(<Probe onState={(s) => captured.push(s)} />);
    act(() => captured[captured.length - 1].updateFilter('priceRange', '0-100'));
    expect(captured[captured.length - 1].filters.priceRange).toBe('0-100');
  });

  test('updateSort sets sortBy', () => {
    const captured: Captured[] = [];
    render(<Probe onState={(s) => captured.push(s)} />);
    act(() => captured[captured.length - 1].updateSort('price_asc'));
    expect(captured[captured.length - 1].sortBy).toBe('price_asc');
  });

  test('resetFilters clears everything', () => {
    const captured: Captured[] = [];
    render(<Probe onState={(s) => captured.push(s)} />);
    act(() => {
      const s = captured[captured.length - 1];
      s.updateFilter('search', 'x');
      s.updateFilter('priceRange', '0-100');
      s.updateSort('price_asc');
    });
    act(() => captured[captured.length - 1].resetFilters());
    const s = captured[captured.length - 1];
    expect(s.rawFilters).toEqual({ search: '', priceRange: '', minRating: '' });
    expect(s.sortBy).toBe('');
    expect(s.hasActiveFilters).toBe(false);
  });
});
