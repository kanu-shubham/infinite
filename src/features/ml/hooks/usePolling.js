import { useEffect, useRef } from "react";

import { POLL_INTERVAL_MS } from "../constants";

/**
 * Calls `callback` on an interval while `active` is true.
 *
 * The callback is held in a ref so a new closure each render does not restart
 * the timer — otherwise a poll that triggers a re-render would reset its own
 * interval and, at worst, never fire.
 */
export default function usePolling(callback, active, interval = POLL_INTERVAL_MS) {
  const callbackRef = useRef(callback);

  useEffect(() => {
    callbackRef.current = callback;
  }, [callback]);

  useEffect(() => {
    if (!active) return undefined;

    const timer = setInterval(() => callbackRef.current(), interval);
    return () => clearInterval(timer);
  }, [active, interval]);
}
