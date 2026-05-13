import { fetchHotels, fetchAllHotels } from './hotelService';
import { mockHotels } from '../data/mockHotels';
import { HOTELS } from '../constants';

beforeEach(() => {
  jest.useFakeTimers();
});
afterEach(() => {
  jest.useRealTimers();
});

async function flush<T>(p: Promise<T>): Promise<T> {
  jest.advanceTimersByTime(HOTELS.simulatedDelay);
  return p;
}

describe('hotelService — fetchAllHotels', () => {
  test('returns every mock hotel with no filters', async () => {
    const result = await flush(fetchAllHotels());
    expect(result).toHaveLength(mockHotels.length);
  });

  test('filters by price range', async () => {
    const result = await flush(fetchAllHotels({ filters: { priceRange: '0-100' } }));
    expect(result.every((h) => h.price <= 100)).toBe(true);
  });

  test('filters by minimum rating', async () => {
    const result = await flush(fetchAllHotels({ filters: { minRating: '4.5' } }));
    expect(result.every((h) => h.rating >= 4.5)).toBe(true);
  });

  test('search matches name OR location, case-insensitive', async () => {
    const result = await flush(fetchAllHotels({ filters: { search: 'GRAND' } }));
    expect(result.length).toBeGreaterThan(0);
    expect(
      result.every(
        (h) =>
          h.name.toLowerCase().includes('grand') ||
          h.location.toLowerCase().includes('grand'),
      ),
    ).toBe(true);
  });

  test('sorts price ascending', async () => {
    const result = await flush(fetchAllHotels({ sortBy: 'price_asc' }));
    for (let i = 1; i < result.length; i++) {
      expect(result[i].price).toBeGreaterThanOrEqual(result[i - 1].price);
    }
  });

  test('sorts rating descending', async () => {
    const result = await flush(fetchAllHotels({ sortBy: 'rating_desc' }));
    for (let i = 1; i < result.length; i++) {
      expect(result[i].rating).toBeLessThanOrEqual(result[i - 1].rating);
    }
  });
});

describe('hotelService — fetchHotels (paginated)', () => {
  test('page 1 returns pageSize items', async () => {
    const r = await flush(fetchHotels({ page: 1 }));
    expect(r.data.length).toBeLessThanOrEqual(HOTELS.pageSize);
    expect(r.total).toBe(mockHotels.length);
    expect(r.hasMore).toBe(r.data.length < r.total);
  });

  test('page 2 returns 2× pageSize items (cumulative)', async () => {
    const r = await flush(fetchHotels({ page: 2 }));
    expect(r.data.length).toBeLessThanOrEqual(HOTELS.pageSize * 2);
  });

  test('hasMore=false when all items are returned', async () => {
    const lastPage = Math.ceil(mockHotels.length / HOTELS.pageSize);
    const r = await flush(fetchHotels({ page: lastPage }));
    expect(r.hasMore).toBe(false);
    expect(r.data.length).toBe(mockHotels.length);
  });
});
