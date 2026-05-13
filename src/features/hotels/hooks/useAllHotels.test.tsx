import React from 'react';
import { act, render } from '@testing-library/react';
import useAllHotels from './useAllHotels';
import { HOTELS } from '../constants';
import { mockHotels } from '../data/mockHotels';

beforeEach(() => jest.useFakeTimers());
afterEach(() => jest.useRealTimers());

type Captured = ReturnType<typeof useAllHotels>;

function Probe({ onState }: { onState: (s: Captured) => void }): null {
  const state = useAllHotels({ filters: {}, sortBy: '' });
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
