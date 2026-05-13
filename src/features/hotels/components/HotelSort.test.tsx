import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import HotelSort from './HotelSort';

describe('<HotelSort />', () => {
  test('count is in a polite live region', () => {
    const { container } = render(
      <HotelSort value="" onChange={() => {}} totalCount={42} />,
    );
    const count = container.querySelector('.hotel-sort__count');
    expect(count).toHaveAttribute('aria-live', 'polite');
    expect(count).toHaveAttribute('aria-atomic', 'true');
    expect(count).toHaveTextContent(/42 hotels found/i);
  });

  test('singular noun for 1 hotel', () => {
    render(<HotelSort value="" onChange={() => {}} totalCount={1} />);
    expect(screen.getByText(/1 hotel found/i)).toBeInTheDocument();
  });

  test('change emits onChange with the new sort key', () => {
    const onChange = jest.fn();
    render(<HotelSort value="" onChange={onChange} totalCount={5} />);
    fireEvent.change(screen.getByLabelText(/sort by/i), { target: { value: 'price_asc' } });
    expect(onChange).toHaveBeenCalledWith('price_asc');
  });
});
