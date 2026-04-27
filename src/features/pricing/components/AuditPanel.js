import React, { useEffect, useState } from "react";
import { pricingClient } from "../services/pricingClient";

export default function AuditPanel({ refreshKey }) {
  const [entries, setEntries] = useState([]);
  const [drift, setDrift] = useState(null);
  const [evalResult, setEvalResult] = useState(null);
  const [uplift, setUplift] = useState(0.05);

  useEffect(() => {
    const ctrl = new AbortController();
    Promise.all([
      pricingClient.audit(20, ctrl.signal).catch(() => ({ entries: [] })),
      pricingClient.drift(ctrl.signal).catch(() => null),
    ]).then(([audit, driftData]) => {
      setEntries(audit.entries);
      setDrift(driftData);
    });
    return () => ctrl.abort();
  }, [refreshKey]);

  const runEval = async () => {
    const r = await pricingClient.evaluate(uplift);
    setEvalResult(r);
  };

  return (
    <div className="audit-panel">
      <div className="audit-row">
        <div className="audit-block">
          <h3>Drift monitor (PSI on price)</h3>
          {drift ? (
            <div className={`drift-tag drift-${drift.status}`}>
              {drift.status} {drift.psi != null && `· ψ=${drift.psi.toFixed(3)}`}
            </div>
          ) : (
            <div className="muted">no data</div>
          )}
        </div>
        <div className="audit-block">
          <h3>Counterfactual eval</h3>
          <div className="eval-controls">
            <label>
              uplift {Math.round(uplift * 100)}%
              <input
                type="range"
                min="-30"
                max="30"
                value={Math.round(uplift * 100)}
                onChange={(e) => setUplift(Number(e.target.value) / 100)}
              />
            </label>
            <button type="button" onClick={runEval}>
              Estimate value
            </button>
          </div>
          {evalResult && (
            <div className="eval-result">
              <div>IPS: ${evalResult.ips.value.toFixed(2)} (ESS {evalResult.ips.ess.toFixed(1)})</div>
              <div>SNIPS: ${evalResult.snips.value.toFixed(2)}</div>
            </div>
          )}
        </div>
      </div>

      <div className="audit-block">
        <h3>Recent decisions</h3>
        <div className="audit-table" role="table" aria-label="Recent pricing decisions">
          <div className="audit-tr audit-th" role="row">
            <span role="columnheader">hotel</span>
            <span role="columnheader">strategy</span>
            <span role="columnheader">base</span>
            <span role="columnheader">proposed</span>
            <span role="columnheader">final</span>
            <span role="columnheader">guardrails</span>
          </div>
          {entries.map((e) => (
            <div key={e.decisionId} className="audit-tr" role="row">
              <span>{e.hotelId}</span>
              <span>{e.strategy}</span>
              <span>${e.basePrice.toFixed(0)}</span>
              <span>${e.proposedPrice.toFixed(2)}</span>
              <span className="strong">${e.finalPrice.toFixed(2)}</span>
              <span className="muted">
                {e.adjustments.length === 0
                  ? "—"
                  : e.adjustments.map((a) => a.reason).join(", ")}
              </span>
            </div>
          ))}
          {entries.length === 0 && <div className="muted">no decisions yet</div>}
        </div>
      </div>
    </div>
  );
}
