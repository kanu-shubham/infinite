const BASE =
  process.env.REACT_APP_PRICING_API ||
  (typeof window !== "undefined" && window.location.hostname === "localhost"
    ? "http://localhost:4000"
    : "");

async function request(path, { method = "GET", body, signal } = {}) {
  const init = { method, signal, headers: {} };
  if (body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json();
}

export const pricingClient = {
  hotels: (signal) => request("/api/pricing/hotels", { signal }),
  quote: (payload, signal) =>
    request("/api/pricing/quote", { method: "POST", body: payload, signal }),
  feedback: (payload, signal) =>
    request("/api/pricing/feedback", { method: "POST", body: payload, signal }),
  explain: (hotelId, ctx = {}, signal) => {
    const qs = new URLSearchParams(
      Object.fromEntries(Object.entries(ctx).filter(([, v]) => v != null))
    ).toString();
    return request(`/api/pricing/explain/${hotelId}${qs ? `?${qs}` : ""}`, { signal });
  },
  elasticity: (hotelId, signal) =>
    request(`/api/pricing/elasticity/${hotelId}`, { signal }),
  audit: (limit = 30, signal) =>
    request(`/api/pricing/audit?limit=${limit}`, { signal }),
  evaluate: (uplift, signal) =>
    request("/api/pricing/evaluate", { method: "POST", body: { uplift }, signal }),
  drift: (signal) => request("/api/pricing/drift", { signal }),
  retrain: (signal) => request("/api/pricing/train", { method: "POST", signal }),
};
