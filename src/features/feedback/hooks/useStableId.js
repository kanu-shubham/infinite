import { useRef } from 'react';

let counter = 0;
const next = () => `fb-${++counter}`;

/**
 * Stable per-instance id. React 18's `useId` would be ideal but this
 * project targets React 17, so we mint our own. Stable across re-renders;
 * not collision-free across server/client hydration (we don't SSR here).
 */
export function useStableId(prefix = 'fb') {
  const ref = useRef(null);
  if (ref.current === null) ref.current = `${prefix}-${next()}`;
  return ref.current;
}
