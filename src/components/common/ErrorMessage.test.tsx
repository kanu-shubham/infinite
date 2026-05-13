import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import ErrorMessage from './ErrorMessage';

describe('<ErrorMessage />', () => {
  test('renders an alert with the message', () => {
    render(<ErrorMessage message="Something broke" />);
    const alert = screen.getByRole('alert');
    expect(alert).toHaveTextContent('Something broke');
  });

  test('omits the retry button when no onRetry is given', () => {
    render(<ErrorMessage message="Boom" />);
    expect(screen.queryByRole('button', { name: /try again/i })).toBeNull();
  });

  test('invokes onRetry when the button is clicked', () => {
    const onRetry = jest.fn();
    render(<ErrorMessage message="Boom" onRetry={onRetry} />);
    fireEvent.click(screen.getByRole('button', { name: /try again/i }));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });
});
