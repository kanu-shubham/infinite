import React, { useEffect, useMemo, useState } from "react";
import { pricingClient } from "./services/pricingClient";
import { useHotels, usePricingExplainer } from "./hooks/usePricing";
import DemandCurve from "./components/DemandCurve";
import PriceQuote from "./components/PriceQuote";
import AuditPanel from "./components/AuditPanel";
import "./PricingDashboard.css";

const DEFAULT_USER = "demo-user";

export default function PricingDashboard() {
  const { hotels, loading, error } = useHotels();
  const [hotelId, setHotelId] = useState(null);
  const [nights, setNights] = useState(2);
  const [leadTimeDays, setLeadTimeDays] = useState(7);
  const [quote, setQuote] = useState(null);
  const [busy, setBusy] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const [errorMsg, setErrorMsg] = useState(null);

  useEffect(() => {
    if (!hotelId && hotels.length) setHotelId(hotels[0].id);
  }, [hotels, hotelId]);

  const context = useMemo(
    () => ({ nights, leadTimeDays, checkIn: new Date().toISOString().slice(0, 10) }),
    [nights, leadTimeDays]
  );

  const explainer = usePricingExplainer(hotelId, context);

  const requestQuote = async () => {
    if (hotelId == null) return;
    setBusy(true);
    setErrorMsg(null);
    try {
      const q = await pricingClient.quote({ hotelId, userId: DEFAULT_USER, context });
      setQuote(q);
      setRefreshKey((k) => k + 1);
    } catch (err) {
      setErrorMsg(err.message);
    } finally {
      setBusy(false);
    }
  };

  const recordOutcome = async (booked) => {
    if (!quote) return;
    setBusy(true);
    setErrorMsg(null);
    try {
      await pricingClient.feedback({
        decisionId: quote.decisionId,
        hotelId,
        price: quote.finalPrice,
        booked,
        units: booked ? 1 : 0,
        context,
      });
      explainer.refresh();
      setRefreshKey((k) => k + 1);
    } catch (err) {
      setErrorMsg(err.message);
    } finally {
      setBusy(false);
    }
  };

  const retrain = async () => {
    setBusy(true);
    setErrorMsg(null);
    try {
      await pricingClient.retrain();
      explainer.refresh();
      setRefreshKey((k) => k + 1);
    } catch (err) {
      setErrorMsg(err.message);
    } finally {
      setBusy(false);
    }
  };

  if (loading) return <div className="pd-empty">Loading hotels…</div>;
  if (error) {
    return (
      <div className="pd-error">
        Could not reach pricing API at {(process.env.REACT_APP_PRICING_API || "http://localhost:4000")}.
        Start it with <code>npm run pricing</code>.
      </div>
    );
  }

  const selectedHotel = hotels.find((h) => h.id === hotelId);

  return (
    <div className="pricing-dashboard">
      <header className="pd-header">
        <h1>Dynamic Pricing Console</h1>
        <p className="pd-sub">
          Bayesian demand model + Thompson-sampling bandit + policy guardrails.
        </p>
      </header>

      <section className="pd-controls">
        <label>
          Hotel
          <select value={hotelId ?? ""} onChange={(e) => setHotelId(Number(e.target.value))}>
            {hotels.map((h) => (
              <option key={h.id} value={h.id}>
                {h.name} — {h.location} (base ${h.basePrice})
              </option>
            ))}
          </select>
        </label>
        <label>
          Nights
          <input
            type="number"
            min={1}
            max={14}
            value={nights}
            onChange={(e) => setNights(Math.max(1, Number(e.target.value) || 1))}
          />
        </label>
        <label>
          Lead time (days)
          <input
            type="number"
            min={0}
            max={365}
            value={leadTimeDays}
            onChange={(e) => setLeadTimeDays(Math.max(0, Number(e.target.value) || 0))}
          />
        </label>
        <button type="button" onClick={requestQuote} disabled={busy || hotelId == null}>
          Get price
        </button>
        <button type="button" onClick={retrain} disabled={busy} className="secondary">
          Retrain models
        </button>
      </section>

      {errorMsg && <div className="pd-error">{errorMsg}</div>}

      <section className="pd-main">
        <div className="pd-card">
          <h2>Quote</h2>
          {quote ? (
            <PriceQuote quote={quote} onBook={recordOutcome} busy={busy} />
          ) : (
            <div className="muted">Click “Get price” to score the current context.</div>
          )}
        </div>

        <div className="pd-card">
          <h2>Demand curve & elasticity</h2>
          {explainer.data ? (
            <>
              <DemandCurve
                curve={explainer.data.curve}
                finalPrice={quote ? quote.finalPrice : null}
                optimalPrice={quote ? quote.optimalPrice : null}
                basePrice={selectedHotel ? selectedHotel.basePrice : null}
              />
              <div className="elasticity-summary">
                <span>
                  fitted elasticity:{" "}
                  <strong>
                    {explainer.data.fittedElasticity != null
                      ? explainer.data.fittedElasticity.toFixed(2)
                      : "—"}
                  </strong>
                </span>
                <span>
                  point elasticity at base:{" "}
                  <strong>{explainer.data.pointElasticityAtBase.toFixed(2)}</strong>
                </span>
                <span>
                  fit R²:{" "}
                  <strong>
                    {explainer.data.r2 != null ? explainer.data.r2.toFixed(3) : "—"}
                  </strong>
                </span>
              </div>
            </>
          ) : explainer.loading ? (
            <div className="muted">Loading…</div>
          ) : (
            <div className="muted">no data</div>
          )}
        </div>
      </section>

      <section className="pd-card">
        <h2>Operations</h2>
        <AuditPanel refreshKey={refreshKey} />
      </section>
    </div>
  );
}
