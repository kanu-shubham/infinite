import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Loads a resource and keeps the last successful value on screen while a
 * refetch is in flight, so polling never flashes a skeleton over live data.
 *
 * `loader` is called with no arguments; pass a stable useCallback so the
 * effect only re-runs when the request actually changes.
 */
export default function useAsyncResource(loader, { enabled = true } = {}) {
  const [state, setState] = useState({
    data: null,
    isLoading: enabled,
    isRefreshing: false,
    error: null,
  });

  // Guards against a slow earlier response overwriting a newer one.
  const requestIdRef = useRef(0);
  const mountedRef = useRef(true);
  const hasDataRef = useRef(false);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const load = useCallback(() => {
    if (!enabled) return Promise.resolve(null);

    const requestId = ++requestIdRef.current;
    setState((previous) => ({
      ...previous,
      isLoading: !hasDataRef.current,
      isRefreshing: hasDataRef.current,
      error: null,
    }));

    return loader()
      .then((data) => {
        if (!mountedRef.current || requestId !== requestIdRef.current) return null;
        hasDataRef.current = true;
        setState({ data, isLoading: false, isRefreshing: false, error: null });
        return data;
      })
      .catch((error) => {
        if (!mountedRef.current || requestId !== requestIdRef.current) return null;
        setState((previous) => ({
          ...previous,
          isLoading: false,
          isRefreshing: false,
          error: error.message,
        }));
        return null;
      });
  }, [loader, enabled]);

  // A new loader means a different resource, so the previous value has to go —
  // otherwise run A's metrics linger under run B's header until B arrives.
  // Polling reuses the same loader and keeps its data, which is what stops
  // every refresh from flashing a skeleton.
  useEffect(() => {
    hasDataRef.current = false;
    setState({ data: null, isLoading: enabled, isRefreshing: false, error: null });
    load();
  }, [load, enabled]);

  return { ...state, reload: load, setData: (data) => setState((p) => ({ ...p, data })) };
}
