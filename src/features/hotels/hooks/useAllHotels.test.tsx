import React from 'react';
import { act, render } from '@testing-library/react';
import useAllHotels from './useAllHotels';
import { HOTELS } from '../constants';
import { mockHotels } from '../data/mockHotels';
import type { HotelFilters, SortKey } from '../types';

beforeEach(() => jest.useFakeTimers());
afterEach(() => jest.useRealTimers());

type Captured = ReturnType<typeof useAllHotels>;

// Module-scope constants so the props passed to useAllHotels are stable
// across renders — otherwise the effect that watches [filters, sortBy]
// re-fires every render.
const STABLE_FILTERS: Partial<HotelFilters> = {};
const STABLE_SORT: SortKey | '' = '';

function Probe({ onState }: { onState: (s: Captured) => void }): null {
  const state = useAllHotels({ filters: STABLE_FILTERS, sortBy: STABLE_SORT });
  onState(state);
  return null;
}

describe('useAllHotels', () => {
  test('returns every hotel after the simulated delay', async () => {
    const captured: Captured[] = [];
    render(<Probe onState={(s) => captured.push(s)} />);
    expect(captured[0].isLoading).toBe(true);

    await act(async () => { jest.advanceTimersByTime(HOTELS.simulatedDelay); });

    const last = captured[captured.length - 1];
    expect(last.isLoading).toBe(false);
    expect(last.hotels).toHaveLength(mockHotels.length);
    expect(last.error).toBeNull();
  });
});
