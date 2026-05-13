import React from 'react';
import { render, act } from '@testing-library/react';
import useDebounce from './useDebounce';

beforeEach(() => jest.useFakeTimers());
afterEach(() => jest.useRealTimers());

interface ProbeProps<T> {
  value: T;
  delay?: number;
  spy: (v: T) => void;
}

function Probe<T>({ value, delay = 200, spy }: ProbeProps<T>): JSX.Element {
  const debounced = useDebounce(value, delay);
  spy(debounced);
  return <span>{String(debounced)}</span>;
}

describe('useDebounce', () => {
  test('returns the initial value immediately', () => {
    const spy = jest.fn();
    render(<Probe value="a" spy={spy} />);
    expect(spy).toHaveBeenLastCalledWith('a');
  });

  test('does not update before delay elapses', () => {
    const spy = jest.fn();
    const { rerender } = render(<Probe value="a" spy={spy} />);
    rerender(<Probe value="b" spy={spy} />);
    act(() => { jest.advanceTimersByTime(100); });
    expect(spy).toHaveBeenLastCalledWith('a');
  });

  test('updates after delay elapses', () => {
    const spy = jest.fn();
    const { rerender } = render(<Probe value="a" spy={spy} />);
    rerender(<Probe value="b" spy={spy} />);
    act(() => { jest.advanceTimersByTime(200); });
    expect(spy).toHaveBeenLastCalledWith('b');
  });

  test('resets the timer on rapid updates', () => {
    const spy = jest.fn();
    const { rerender } = render(<Probe value="a" spy={spy} />);
    rerender(<Probe value="b" spy={spy} />);
    act(() => { jest.advanceTimersByTime(100); });
    rerender(<Probe value="c" spy={spy} />);
    act(() => { jest.advanceTimersByTime(100); });
    expect(spy).toHaveBeenLastCalledWith('a');
    act(() => { jest.advanceTimersByTime(100); });
    expect(spy).toHaveBeenLastCalledWith('c');
  });
});
