import { useEffect, useRef } from 'react';

/**
 * Fires `onDismiss` after `delay` ms while `active` is true.
 *
 * Why not setTimeout in a component effect with no extras? Because:
 *   - tests need fake timers,
 *   - the caller may want to cancel early (e.g. user closes the toast),
 *   - StrictMode double-invokes effects in dev, so the cleanup must be tight.
 */
export function useAutoDismiss(onDismiss, { active, delay = 2000 }) {
  const handlerRef = useRef(onDismiss);
  handlerRef.current = onDismiss;

  useEffect(() => {
    if (!active) return undefined;
    const id = window.setTimeout(() => handlerRef.current?.(), delay);
    return () => window.clearTimeout(id);
  }, [active, delay]);
}
