import React from 'react';
import { render, screen } from '@testing-library/react';
import HotelSkeletonGrid from './HotelSkeleton';

describe('<HotelSkeletonGrid />', () => {
  test('renders the requested number of skeleton cards inside a status region', () => {
    const { container } = render(<HotelSkeletonGrid count={5} />);
    expect(screen.getByRole('status', { name: /loading hotels/i })).toBeInTheDocument();
    expect(container.querySelectorAll('.hotel-skeleton')).toHaveLength(5);
  });

  test('defaults to 8 cards', () => {
    const { container } = render(<HotelSkeletonGrid />);
    expect(container.querySelectorAll('.hotel-skeleton')).toHaveLength(8);
  });
});
