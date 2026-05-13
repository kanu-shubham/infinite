import React from 'react';
import { act, render } from '@testing-library/react';
import useHotels from './useHotels';
import { HOTELS } from '../constants';
import { mockHotels } from '../data/mockHotels';

beforeEach(() => jest.useFakeTimers());
afterEach(() => jest.useRealTimers());

type Captured = ReturnType<typeof useHotels>;

function Probe({
  onState,
  filters = {},
  sortBy = '',
}: {
  onState: (s: Captured) => void;
  filters?: Parameters<typeof useHotels>[0]['filters'];
  sortBy?: Parameters<typeof useHotels>[0]['sortBy'];
}): null {
  const state = useHotels({ filters, sortBy });
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
