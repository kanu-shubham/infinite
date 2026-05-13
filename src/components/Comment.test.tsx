import React from 'react';
import { render, screen } from '@testing-library/react';
import Comment from './Comment';

const fixture = {
  postId: 1,
  id: 42,
  name: 'Anon',
  email: 'anon@example.com',
  body: 'Nice post.',
};

describe('<Comment />', () => {
  test('renders id, email, and body', () => {
    render(<Comment comment={fixture} />);
    expect(screen.getByText(/\[42\] anon@example\.com/)).toBeInTheDocument();
    expect(screen.getByText('Nice post.')).toBeInTheDocument();
  });

  test('forwards mesureRef to the underlying li', () => {
    const ref = jest.fn();
    render(<Comment comment={fixture} mesureRef={ref} />);
    expect(ref).toHaveBeenCalled();
    expect(ref.mock.calls[0][0]).toBeInstanceOf(HTMLElement);
  });
});
