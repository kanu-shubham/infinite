import React from 'react';
import { render, screen } from '@testing-library/react';
import CommentList from './CommentList';
import { MockIntersectionObserver } from '../setupTests';

const comments = [
  { postId: 1, id: 1, name: 'a', email: 'a@a', body: 'one' },
  { postId: 1, id: 2, name: 'b', email: 'b@b', body: 'two' },
];

describe('<CommentList />', () => {
  test('renders all comments and a Loading row when isLoading', () => {
    render(<CommentList hasMore comments={comments} isLoading loadMore={() => {}} />);
    expect(screen.getByText('one')).toBeInTheDocument();
    expect(screen.getByText('two')).toBeInTheDocument();
    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });

  test('calls loadMore once the sentinel intersects (when hasMore=true)', () => {
    const loadMore = jest.fn();
    render(<CommentList hasMore comments={comments} isLoading={false} loadMore={loadMore} />);
    MockIntersectionObserver.triggerAll(true);
    expect(loadMore).toHaveBeenCalled();
  });

  test('does not call loadMore when hasMore=false', () => {
    const loadMore = jest.fn();
    render(<CommentList hasMore={false} comments={comments} isLoading={false} loadMore={loadMore} />);
    MockIntersectionObserver.triggerAll(true);
    expect(loadMore).not.toHaveBeenCalled();
  });
});
