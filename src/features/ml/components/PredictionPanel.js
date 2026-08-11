import React, { useCallback, useEffect, useMemo, useState } from "react";

import ErrorMessage from "../../../components/common/ErrorMessage";
import LoadingSpinner from "../../../components/common/LoadingSpinner";
import { useDataset } from "../hooks/useCatalog";
import { fetchSampleRows, predict } from "../services/mlApi";
import "./PredictionPanel.css";

function initialValues(features) {
  return features.reduce(
    (accumulator, feature) => ({ ...accumulator, [feature.name]: feature.default }),
    {}
  );
}

/**
 * The serving half of the app: score a hypothetical booking against a run's
 * registered pipeline. The form is generated from the same feature specs the
 * pipeline was fitted on, so the two cannot drift apart.
 */
export default function PredictionPanel({ run }) {
  const target = run.config.target;
  const { dataset, isLoading, error: datasetError } = useDataset(target);

  const features = useMemo(() => dataset?.features ?? [], [dataset]);
  const [values, setValues] = useState({});
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState({ isBusy: false, error: null });

  useEffect(() => {
    if (features.length) setValues(initialValues(features));
  }, [features]);

  // A prediction belongs to the run that produced it — never let one linger
  // when the user selects a different run.
  useEffect(() => {
    setResult(null);
    setStatus({ isBusy: false, error: null });
  }, [run.run_id]);

  const setValue = useCallback((name, value) => {
    setValues((previous) => ({ ...previous, [name]: value }));
    setResult(null);
  }, []);

  const loadSample = useCallback(async () => {
    setStatus({ isBusy: true, error: null });
    try {
      const { rows } = await fetchSampleRows(target, 1);
      const row = rows[0];
      // Keep only the columns the model actually consumes.
      setValues(
        features.reduce((accumulator, feature) => {
          const value = row[feature.name];
          return {
            ...accumulator,
            [feature.name]: value === null || value === undefined ? feature.default : value,
          };
        }, {})
      );
      setResult(null);
      setStatus({ isBusy: false, error: null });
    } catch (error) {
      setStatus({ isBusy: false, error: error.message });
    }
  }, [features, target]);

  const submit = useCallback(
    async (event) => {
      event.preventDefault();
      setStatus({ isBusy: true, error: null });
      try {
        const response = await predict(run.run_id, [values]);
        setResult(response.predictions[0]);
        setStatus({ isBusy: false, error: null });
      } catch (error) {
        setResult(null);
        setStatus({ isBusy: false, error: error.message });
      }
    },
    [run.run_id, values]
  );

  if (isLoading) return <LoadingSpinner text="Loading feature schema…" />;
  if (datasetError) return <ErrorMessage message={datasetError} />;

  const categorical = features.filter((feature) => feature.kind === "categorical");
  const numeric = features.filter((feature) => feature.kind === "numeric");
  const isClassification = run.config.task === "classification";

  return (
    <section className="run-detail__section">
      <h3 className="run-detail__section-title">Score a booking</h3>

      <div className="predict">
        <form className="predict__form" onSubmit={submit}>
          <div className="predict__toolbar">
            <p className="predict__intro">
              Sent to <code>POST /api/runs/{run.run_id}/predict</code>. Anything you
              leave alone is imputed by the fitted pipeline, exactly as it was at
              training time.
            </p>
            <button
              type="button"
              className="predict__secondary"
              onClick={loadSample}
              disabled={status.isBusy}
            >
              Load a real booking
            </button>
          </div>

          <div className="predict__grid">
            {categorical.map((feature) => (
              <div className="predict__field" key={feature.name}>
                <label className="predict__label" htmlFor={`pf-${feature.name}`}>
                  {feature.label}
                </label>
                <select
                  id={`pf-${feature.name}`}
                  className="predict__control"
                  value={values[feature.name] ?? ""}
                  onChange={(event) => setValue(feature.name, event.target.value)}
                >
                  {feature.options.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </div>
            ))}

            {numeric.map((feature) => (
              <div className="predict__field" key={feature.name}>
                <label className="predict__label" htmlFor={`pf-${feature.name}`}>
                  {feature.label}
                </label>
                <input
                  id={`pf-${feature.name}`}
                  type="number"
                  className="predict__control"
                  value={values[feature.name] ?? ""}
                  min={feature.minimum}
                  max={feature.maximum}
                  step={feature.step}
                  onChange={(event) => {
                    const raw = event.target.value;
                    setValue(feature.name, raw === "" ? null : Number(raw));
                  }}
                />
              </div>
            ))}
          </div>

          <button type="submit" className="predict__submit" disabled={status.isBusy}>
            {status.isBusy ? "Scoring…" : "Predict"}
          </button>
        </form>

        <aside className="predict__result">
          {status.error && <ErrorMessage message={status.error} />}

          {!status.error && !result && (
            <p className="predict__placeholder">
              Adjust the booking and hit Predict — the answer comes from the
              pipeline this run registered, not a re-fit.
            </p>
          )}

          {result && isClassification && (
            <div className="predict__outcome">
              <span className="predict__outcome-label">Prediction</span>
              <span className="predict__outcome-value">{result.label}</span>
              {result.probability !== null && result.probability !== undefined && (
                <>
                  <span className="predict__probability-label">
                    Cancellation probability
                  </span>
                  <div className="predict__meter">
                    <div
                      className="predict__meter-fill"
                      style={{ width: `${result.probability * 100}%` }}
                    />
                  </div>
                  <span className="predict__probability-value">
                    {(result.probability * 100).toFixed(1)}%
                  </span>
                </>
              )}
            </div>
          )}

          {result && !isClassification && (
            <div className="predict__outcome">
              <span className="predict__outcome-label">Predicted nightly rate</span>
              <span className="predict__outcome-value">
                ${result.prediction.toFixed(2)}
              </span>
            </div>
          )}
        </aside>
      </div>
    </section>
  );
}
