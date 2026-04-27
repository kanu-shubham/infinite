import React from "react";

export default function PriceQuote({ quote, onBook, busy }) {
  if (!quote) return null;
  const delta = quote.basePrice ? ((quote.finalPrice - quote.basePrice) / quote.basePrice) * 100 : 0;
  const deltaClass = delta > 0 ? "up" : delta < 0 ? "down" : "flat";

  return (
    <div className="price-quote">
      <div className="pq-row pq-headline">
        <span className="pq-price">${quote.finalPrice.toFixed(0)}</span>
        <span className={`pq-delta pq-delta-${deltaClass}`}>
          {delta >= 0 ? "+" : ""}
          {delta.toFixed(1)}% vs base ${quote.basePrice.toFixed(0)}
        </span>
        <span className="pq-strategy">{quote.strategy}</span>
      </div>

      <div className="pq-grid">
        <Metric label="proposed (model)" value={`$${quote.proposedPrice.toFixed(0)}`} />
        <Metric label="optimizer (greedy)" value={`$${quote.optimalPrice.toFixed(0)}`} />
        <Metric
          label="P(book)"
          value={`${(quote.conversionProb * 100).toFixed(1)}%`}
        />
        <Metric label="exp. units" value={quote.expectedUnits.toFixed(2)} />
        <Metric label="exp. revenue" value={`$${quote.expectedRevenue.toFixed(0)}`} />
        <Metric label="propensity" value={quote.propensity.toFixed(3)} />
      </div>

      {quote.adjustments.length > 0 && (
        <div className="pq-adjustments">
          <div className="pq-section-title">Guardrails fired</div>
          <ul>
            {quote.adjustments.map((a, i) => (
              <li key={i}>
                <code>{a.reason}</code>: ${Number(a.from).toFixed(2)} → ${Number(a.to).toFixed(2)}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="pq-actions">
        <button type="button" onClick={() => onBook(true)} disabled={busy}>
          Simulate booking
        </button>
        <button type="button" onClick={() => onBook(false)} disabled={busy} className="secondary">
          Simulate skip
        </button>
      </div>
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div className="pq-metric">
      <div className="pq-metric-label">{label}</div>
      <div className="pq-metric-value">{value}</div>
    </div>
  );
}
