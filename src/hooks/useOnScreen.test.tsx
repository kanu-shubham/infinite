import React from 'react';
import { render } from '@testing-library/react';
import useOnScreen from './useOnScreen';
import { MockIntersectionObserver } from '../setupTests';

function Probe({ onState }: { onState: (s: { isIntersecting: boolean }) => void }) {
  const { measureRef, isIntersecting } = useOnScreen();
  onState({ isIntersecting });
  return <div data-testid="el" ref={(el) => measureRef(el)} />;
}

describe('useOnScreen', () => {
  test('reports intersecting=true after IntersectionObserver fires', () => {
    const states: boolean[] = [];
    render(<Probe onState={(s) => states.push(s.isIntersecting)} />);
    expect(states[0]).toBe(false);
    MockIntersectionObserver.triggerAll(true);
    expect(states[states.length - 1]).toBe(true);
  });
});
