import React from 'react';
import { act, fireEvent, render, screen } from '@testing-library/react';
import HotelListingPage from './HotelListingPage';
import { HOTELS } from './constants';

beforeEach(() => jest.useFakeTimers());
afterEach(() => {
  act(() => { jest.runOnlyPendingTimers(); });
  jest.useRealTimers();
});

async function flushFetches(): Promise<void> {
  // Both useAllHotels and useHotels race their fetch on mount.
  await act(async () => { jest.advanceTimersByTime(HOTELS.simulatedDelay); });
}

describe('<HotelListingPage /> integration', () => {
  test('renders the heading, filters landmark, and tablist', async () => {
    render(<HotelListingPage />);
    expect(screen.getByRole('heading', { name: /find your perfect stay/i })).toBeInTheDocument();
    expect(screen.getByRole('search', { name: /filter hotels/i })).toBeInTheDocument();
    expect(screen.getByRole('tablist', { name: /list rendering mode/i })).toBeInTheDocument();
    await flushFetches();
  });

  test('tabs reflect aria-selected/aria-pressed when active', async () => {
    render(<HotelListingPage />);
    await flushFetches();
    const virtualTab = screen.getByRole('tab', { name: /virtualized/i });
    const standardTab = screen.getByRole('tab', { name: /standard/i });
    expect(virtualTab).toHaveAttribute('aria-selected', 'true');
    expect(standardTab).toHaveAttribute('aria-selected', 'false');

    fireEvent.click(standardTab);
    expect(standardTab).toHaveAttribute('aria-selected', 'true');
    expect(virtualTab).toHaveAttribute('aria-selected', 'false');
  });

  test('switching to Standard renders a tabpanel and a result count', async () => {
    render(<HotelListingPage />);
    await flushFetches();
    fireEvent.click(screen.getByRole('tab', { name: /standard/i }));
    await flushFetches();
    expect(screen.getAllByRole('tabpanel').length).toBeGreaterThan(0);
    // Both the sr-only live region and the visible count match; both
    // appearing is the desired behaviour — assert at least one is rendered.
    expect(screen.getAllByText(/hotels found/i).length).toBeGreaterThanOrEqual(1);
  });

  test('strategy toggle exposes aria-pressed', async () => {
    render(<HotelListingPage />);
    await flushFetches();
    const container = screen.getByRole('button', { name: /container scroll/i });
    const win = screen.getByRole('button', { name: /window scroll/i });
    expect(container).toHaveAttribute('aria-pressed', 'true');
    expect(win).toHaveAttribute('aria-pressed', 'false');
    fireEvent.click(win);
    expect(win).toHaveAttribute('aria-pressed', 'true');
    expect(container).toHaveAttribute('aria-pressed', 'false');
  });
});
