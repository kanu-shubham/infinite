import React from 'react';
import { render } from '@testing-library/react';
import useInfiniteScroll from './useInfiniteScroll';
import { MockIntersectionObserver } from '../../../setupTests';

function Probe({ onLoadMore, enabled = true }: { onLoadMore: () => void; enabled?: boolean }) {
  const ref = useInfiniteScroll(onLoadMore, { enabled });
  return <div data-testid="sentinel" ref={ref as unknown as React.Ref<HTMLDivElement>} />;
}

describe('useInfiniteScroll', () => {
  test('calls onLoadMore when sentinel intersects', () => {
    const onLoadMore = jest.fn();
    render(<Probe onLoadMore={onLoadMore} />);
    MockIntersectionObserver.triggerAll(true);
    expect(onLoadMore).toHaveBeenCalledTimes(1);
  });

  test('does not call onLoadMore when disabled', () => {
    const onLoadMore = jest.fn();
    render(<Probe onLoadMore={onLoadMore} enabled={false} />);
    MockIntersectionObserver.triggerAll(true);
    expect(onLoadMore).not.toHaveBeenCalled();
  });

  test('ignores non-intersecting events', () => {
    const onLoadMore = jest.fn();
    render(<Probe onLoadMore={onLoadMore} />);
    MockIntersectionObserver.triggerAll(false);
    expect(onLoadMore).not.toHaveBeenCalled();
  });
});
