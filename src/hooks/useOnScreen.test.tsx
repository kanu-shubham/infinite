import React from 'react';
import { act, render } from '@testing-library/react';
import useOnScreen from './useOnScreen';
import { MockIntersectionObserver } from '../setupTests';

function Probe({ onState }: { onState: (s: { isIntersecting: boolean }) => void }) {
  const { measureRef, isIntersecting } = useOnScreen();
  onState({ isIntersecting });
  // measureRef is a stable useCallback — pass it directly so React only
  // invokes it on actual node attach/detach, not on every render.
  return <div data-testid="el" ref={measureRef as unknown as React.LegacyRef<HTMLDivElement>} />;
}

describe('useOnScreen', () => {
  test('reports intersecting=true after IntersectionObserver fires', () => {
    const states: boolean[] = [];
    render(<Probe onState={(s) => states.push(s.isIntersecting)} />);
    expect(states[0]).toBe(false);
    act(() => { MockIntersectionObserver.triggerAll(true); });
    expect(states[states.length - 1]).toBe(true);
  });
});
