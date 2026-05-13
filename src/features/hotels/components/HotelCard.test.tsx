import React from 'react';
import { render, screen } from '@testing-library/react';
import HotelCard from './HotelCard';
import type { Hotel } from '../types';

const hotel: Hotel = {
  id: 1,
  name: 'Test Hotel',
  location: 'Amsterdam, NL',
  price: 199,
  rating: 4.6,
  reviewCount: 128,
  image: 'https://example.com/x.jpg',
  amenities: ['WiFi', 'Pool', 'Gym', 'Spa', 'Bar', 'Parking'],
  description: 'A nice place.',
};

describe('<HotelCard /> a11y + content', () => {
  test('uses an article landmark with a fully descriptive aria-label', () => {
    render(<HotelCard hotel={hotel} />);
    const article = screen.getByRole('article', { name: /Test Hotel/i });
    expect(article).toHaveAttribute(
      'aria-label',
      'Test Hotel, Amsterdam, NL, $199 per night, rated 4.6 out of 5',
    );
  });

  test('image alt describes the entity in context', () => {
    render(<HotelCard hotel={hotel} />);
    expect(screen.getByRole('img', { name: /Exterior view of Test Hotel in Amsterdam, NL/i }))
      .toBeInTheDocument();
  });

  test('star rating exposes accessible name', () => {
    render(<HotelCard hotel={hotel} />);
    expect(screen.getByRole('img', { name: /4\.6 out of 5 stars/i })).toBeInTheDocument();
  });

  test('shows the first 4 amenities and a "+more" pill for the rest', () => {
    render(<HotelCard hotel={hotel} />);
    ['WiFi', 'Pool', 'Gym', 'Spa'].forEach((a) => {
      expect(screen.getByText(a)).toBeInTheDocument();
    });
    expect(screen.getByText(/\+2 more/)).toBeInTheDocument();
  });

  test('price badge is hidden from assistive tech (already in aria-label)', () => {
    const { container } = render(<HotelCard hotel={hotel} />);
    const price = container.querySelector('.hotel-card__price');
    expect(price).toHaveAttribute('aria-hidden', 'true');
  });
});
