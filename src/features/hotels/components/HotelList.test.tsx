import React from 'react';
import { render, screen } from '@testing-library/react';
import HotelList from './HotelList';
import type { Hotel } from '../types';

const makeHotel = (id: number): Hotel => ({
  id,
  name: `Hotel ${id}`,
  location: 'Test Location',
  price: 100 + id,
  rating: 4,
  reviewCount: 10,
  image: 'x.jpg',
  amenities: ['WiFi'],
  description: '',
});

describe('<HotelList />', () => {
  test('shows skeleton grid while loading with no hotels yet', () => {
    render(<HotelList hotels={[]} isLoading hasMore sentinelRef={() => {}} />);
    expect(screen.getByRole('status', { name: /loading hotels/i })).toBeInTheDocument();
  });

  test('shows empty state when not loading and no hotels', () => {
    render(<HotelList hotels={[]} isLoading={false} hasMore={false} sentinelRef={() => {}} />);
    expect(screen.getByText(/no hotels found/i)).toBeInTheDocument();
  });

  test('renders hotels inside an explicit list', () => {
    const hotels = [makeHotel(1), makeHotel(2)];
    render(<HotelList hotels={hotels} isLoading={false} hasMore={false} sentinelRef={() => {}} />);
    const list = screen.getByRole('list');
    expect(list).toBeInTheDocument();
    expect(screen.getAllByRole('listitem')).toHaveLength(2);
  });

  test('renders sentinel when hasMore=true', () => {
    const sentinelRef = jest.fn();
    render(
      <HotelList
        hotels={[makeHotel(1)]}
        isLoading={false}
        hasMore
        sentinelRef={sentinelRef}
      />,
    );
    expect(sentinelRef).toHaveBeenCalled();
  });

  test('shows "all results" message when hasMore=false and there are hotels', () => {
    render(
      <HotelList
        hotels={[makeHotel(1)]}
        isLoading={false}
        hasMore={false}
        sentinelRef={() => {}}
      />,
    );
    expect(screen.getByText(/you've seen all results/i)).toBeInTheDocument();
  });
});
