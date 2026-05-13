import React from 'react';
import { act, render } from '@testing-library/react';
import useHotels from './useHotels';
import { HOTELS } from '../constants';
import { mockHotels } from '../data/mockHotels';
import type { HotelFilters, SortKey } from '../types';

beforeEach(() => jest.useFakeTimers());
afterEach(() => jest.useRealTimers());

type Captured = ReturnType<typeof useHotels>;

// IMPORTANT: filters / sortBy are stored in module-scope refs so the
// values passed to useHotels are stable across re-renders. Without this
// stability the effect that watches [filters, sortBy] would re-fire every
// render → dispatch RESET → state change → render → loop.
const STABLE_FILTERS: Partial<HotelFilters> = {};
const STABLE_SORT: SortKey | '' = '';

function Probe({ onState }: { onState: (s: Captured) => void }): null {
  const state = useHotels({ filters: STABLE_FILTERS, sortBy: STABLE_SORT });
  onState(state);
  return null;
}

describe('useHotels (paginated)', () => {
  test('fetches first page on mount', async () => {
    const captured: Captured[] = [];
    render(<Probe onState={(s) => captured.push(s)} />);
    expect(captured[0].isLoading).toBe(true);

    await act(async () => {
      jest.advanceTimersByTime(HOTELS.simulatedDelay);
    });

    const last = captured[captured.length - 1];
    expect(last.isLoading).toBe(false);
    expect(last.hotels.length).toBe(HOTELS.pageSize);
    expect(last.totalCount).toBe(mockHotels.length);
    expect(last.hasMore).toBe(true);
  });

  test('loadMore appends the next page', async () => {
    const captured: Captured[] = [];
    render(<Probe onState={(s) => captured.push(s)} />);
    await act(async () => { jest.advanceTimersByTime(HOTELS.simulatedDelay); });

    act(() => captured[captured.length - 1].loadMore());
    await act(async () => { jest.advanceTimersByTime(HOTELS.simulatedDelay); });

    const last = captured[captured.length - 1];
    expect(last.hotels.length).toBe(HOTELS.pageSize * 2);
  });
});
