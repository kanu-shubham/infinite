import { useCallback, useEffect, useRef, useState } from "react";
import { pricingClient } from "../services/pricingClient";

export function useHotels() {
  const [hotels, setHotels] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const ctrl = new AbortController();
    pricingClient
      .hotels(ctrl.signal)
      .then((res) => {
        setHotels(res.hotels);
        setLoading(false);
      })
      .catch((err) => {
        if (err.name !== "AbortError") {
          setError(err);
          setLoading(false);
        }
      });
    return () => ctrl.abort();
  }, []);

  return { hotels, loading, error };
}

export function usePricingExplainer(hotelId, context) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const requestId = useRef(0);

  const refresh = useCallback(() => {
    if (hotelId == null) return;
    const id = ++requestId.current;
    setLoading(true);
    pricingClient
      .explain(hotelId, context)
      .then((res) => {
        if (id !== requestId.current) return;
        setData(res);
        setError(null);
        setLoading(false);
      })
      .catch((err) => {
        if (id !== requestId.current) return;
        setError(err);
        setLoading(false);
      });
  }, [hotelId, JSON.stringify(context)]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { data, loading, error, refresh };
}
