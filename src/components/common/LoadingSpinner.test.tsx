import React from 'react';
import { render, screen } from '@testing-library/react';
import LoadingSpinner from './LoadingSpinner';

describe('<LoadingSpinner />', () => {
  test('exposes role=status with the loading text as accessible name', () => {
    render(<LoadingSpinner text="Fetching hotels…" />);
    const status = screen.getByRole('status');
    expect(status).toHaveAttribute('aria-live', 'polite');
    expect(status).toHaveAttribute('aria-label', 'Fetching hotels…');
  });

  test('falls back to "Loading" when text is omitted', () => {
    render(<LoadingSpinner text="" />);
    expect(screen.getByRole('status')).toHaveAttribute('aria-label', 'Loading');
  });

  test('applies size modifier class', () => {
    const { container } = render(<LoadingSpinner size="large" />);
    expect(container.firstChild).toHaveClass('loading-spinner--large');
  });
});
