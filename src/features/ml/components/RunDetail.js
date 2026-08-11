import React from "react";

import BarList from "./charts/BarList";
import ChartCard from "./charts/ChartCard";
import ConfusionMatrix from "./charts/ConfusionMatrix";
import PipelineDiagram from "./PipelineDiagram";
import PredictionScatter from "./charts/PredictionScatter";
import RocCurve from "./charts/RocCurve";
import StatusPill from "./StatusPill";
import { METRIC_ORDER } from "../constants";
import {
  formatCount,
  formatDuration,
  formatMetric,
  formatTimestamp,
  humanize,
  metricLabel,
  splitEncodedFeature,
} from "../utils/format";
import "./RunDetail.css";

function MetricTiles({ task, scores, headlineKey }) {
  const order = METRIC_ORDER[task].filter((key) => scores[key] !== undefined);

  return (
    <div className="run-detail__metrics">
      {order.map((key) => (
        <div
          key={key}
          className={`metric-tile${key === headlineKey ? " metric-tile--headline" : ""}`}
        >
          <span className="metric-tile__label">{metricLabel(key)}</span>
          <span className="metric-tile__value">{formatMetric(key, scores[key])}</span>
          {key === headlineKey && (
            <span className="metric-tile__flag">headline metric</span>
          )}
        </div>
      ))}
    </div>
  );
}

function ConfigSummary({ run }) {
  const { config } = run;
  const entries = [
    ["Estimator", config.model_label],
    ["Target", `${config.target} (${config.task})`],
    ["Hold-out", `${Math.round(config.test_size * 100)}%`],
    ["Dataset rows", formatCount(config.dataset_rows)],
    ["Seed", config.seed],
    ["Cross-validation", config.cv_folds ? `${config.cv_folds}-fold` : "off"],
  ];

  const hyperparameters = Object.entries(run.resolved_hyperparameters || {});

  return (
    <div className="run-detail__config">
      <dl className="run-detail__config-list">
        {entries.map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>

      {hyperparameters.length > 0 && (
        <>
          <h4 className="run-detail__config-heading">Resolved hyperparameters</h4>
          <dl className="run-detail__config-list">
            {hyperparameters.map(([name, value]) => (
              <div key={name}>
                <dt>{humanize(name)}</dt>
                <dd>{String(value)}</dd>
              </div>
            ))}
          </dl>
        </>
      )}
    </div>
  );
}

function LogConsole({ logs }) {
  if (!logs.length) return null;

  return (
    <details className="run-detail__logs">
      <summary>Run log ({logs.length} entries)</summary>
      <ol className="log-console">
        {logs.map((entry, index) => (
          <li key={index} className={`log-console__row log-console__row--${entry.level}`}>
            <span className="log-console__time">{formatTimestamp(entry.ts)}</span>
            <span className="log-console__message">{entry.message}</span>
          </li>
        ))}
      </ol>
    </details>
  );
}

/**
 * Everything one run produced. While the run is in flight this is just the
 * pipeline diagram; the metric and chart sections appear as the backend
 * fills them in.
 */
export default function RunDetail({ run, sourceColumns, children }) {
  const isFinished = run.status === "succeeded";
  const metrics = run.metrics;

  // For a one-hot column the level is the distinguishing part, so it leads and
  // the source column follows — "Non Refund · deposit type" rather than three
  // rows that all truncate to "Deposit type = …".
  const importanceItems = (run.feature_importances || []).map((entry) => {
    const { base, level } = splitEncodedFeature(entry.feature, sourceColumns);
    return {
      id: entry.feature,
      label: level || base,
      sublabel: level ? base.toLowerCase() : null,
      tableLabel: level ? `${base} = ${level}` : base,
      value: entry.importance,
    };
  });

  const importanceKind = run.feature_importances?.[0]?.kind;

  const cvItems = (run.cross_validation?.scores || []).map((score, index) => ({
    id: `fold-${index}`,
    label: `Fold ${index + 1}`,
    value: score,
  }));

  return (
    <div className="run-detail">
      <header className="run-detail__header">
        <div>
          <h2 className="run-detail__title">{run.name}</h2>
          <p className="run-detail__subtitle">
            <code>{run.run_id}</code>
            {run.duration_ms !== null && ` · ${formatDuration(run.duration_ms)}`}
          </p>
        </div>
        <StatusPill status={run.status} />
      </header>

      {run.status === "failed" && (
        <div className="run-detail__failure" role="alert">
          <strong>
            Failed{run.failed_stage ? ` during ${run.failed_stage}` : ""}
          </strong>
          <span>{run.error}</span>
        </div>
      )}

      <section className="run-detail__section">
        <h3 className="run-detail__section-title">Pipeline</h3>
        <div className="run-detail__panel">
          <PipelineDiagram stages={run.stages} progress={run.progress} />
        </div>
      </section>

      {isFinished && metrics && (
        <>
          <section className="run-detail__section">
            <h3 className="run-detail__section-title">Held-out performance</h3>
            <MetricTiles
              task={run.task}
              scores={metrics.scores}
              headlineKey={metrics.primary}
            />
            {run.dataset && (
              <p className="run-detail__caption">
                Scored on {formatCount(run.dataset.test_rows)} held-out rows ·
                {" "}
                {formatCount(run.dataset.train_rows)} used for training ·
                {" "}
                {run.dataset.raw_features} raw features expanded to{" "}
                {run.dataset.encoded_features} model inputs.
              </p>
            )}
          </section>

          <section className="run-detail__section">
            <h3 className="run-detail__section-title">Diagnostics</h3>
            <div className="run-detail__charts">
              {run.task === "classification" && metrics.roc_curve && (
                <ChartCard
                  title="ROC curve"
                  subtitle={`Area under the curve: ${formatMetric(
                    "roc_auc",
                    metrics.scores.roc_auc
                  )}`}
                  footnote="Further above the chance line means the model separates the two classes better at every threshold."
                  table={{
                    columns: [
                      { key: "fpr", label: "False positive rate", align: "right" },
                      { key: "tpr", label: "True positive rate", align: "right" },
                    ],
                    rows: metrics.roc_curve.map((point, index) => ({
                      id: index,
                      fpr: point.fpr.toFixed(3),
                      tpr: point.tpr.toFixed(3),
                    })),
                  }}
                >
                  <RocCurve points={metrics.roc_curve} auc={metrics.scores.roc_auc} />
                </ChartCard>
              )}

              {run.task === "classification" && metrics.confusion_matrix && (
                <ChartCard
                  title="Confusion matrix"
                  subtitle="Where the held-out predictions landed."
                >
                  <ConfusionMatrix
                    matrix={metrics.confusion_matrix}
                    positiveLabel="Cancelled"
                    negativeLabel="Honoured"
                  />
                </ChartCard>
              )}

              {run.task === "regression" && metrics.scatter && (
                <ChartCard
                  title="Predicted vs actual"
                  subtitle={`Residual spread: ${metrics.residual_summary.p05} to ${metrics.residual_summary.p95} (5th–95th percentile)`}
                  footnote="Vertical distance from the diagonal is the error on that booking."
                  table={{
                    columns: [
                      { key: "actual", label: "Actual", align: "right" },
                      { key: "predicted", label: "Predicted", align: "right" },
                      { key: "error", label: "Error", align: "right" },
                    ],
                    rows: metrics.scatter.map((point, index) => ({
                      id: index,
                      actual: point.actual.toFixed(2),
                      predicted: point.predicted.toFixed(2),
                      error: (point.predicted - point.actual).toFixed(2),
                    })),
                  }}
                >
                  <PredictionScatter points={metrics.scatter} unit="rate (USD)" />
                </ChartCard>
              )}

              <ChartCard
                title="Feature importance"
                subtitle={
                  importanceKind === "permutation"
                    ? "Permutation importance, measured on the training split."
                    : `Relative ${importanceKind || "model"} importance, normalised to sum to 1.`
                }
                footnote="One-hot levels are shown against their source column."
                table={{
                  columns: [
                    { key: "feature", label: "Feature" },
                    { key: "importance", label: "Importance", align: "right" },
                  ],
                  rows: importanceItems.map((item) => ({
                    id: item.id,
                    feature: item.tableLabel,
                    importance: item.value.toFixed(4),
                  })),
                }}
              >
                <BarList
                  items={importanceItems}
                  formatValue={(value) => value.toFixed(3)}
                  emptyMessage="This estimator did not expose importances."
                />
              </ChartCard>

              {run.cross_validation && (
                <ChartCard
                  title="Cross-validation"
                  subtitle={`${run.cross_validation.folds}-fold ${run.cross_validation.scoring} on the training split`}
                  footnote={`Mean ${run.cross_validation.mean} ± ${run.cross_validation.std}. A wide spread means the held-out score is luck-sensitive.`}
                  table={{
                    columns: [
                      { key: "fold", label: "Fold" },
                      { key: "score", label: run.cross_validation.scoring, align: "right" },
                    ],
                    rows: cvItems.map((item, index) => ({
                      id: item.id,
                      fold: `Fold ${index + 1}`,
                      score: item.value.toFixed(4),
                    })),
                  }}
                >
                  <BarList items={cvItems} formatValue={(value) => value.toFixed(3)} />
                </ChartCard>
              )}
            </div>
          </section>

          {children}
        </>
      )}

      <section className="run-detail__section">
        <h3 className="run-detail__section-title">Configuration</h3>
        <div className="run-detail__panel">
          <ConfigSummary run={run} />
          {run.pipeline_steps?.length > 0 && (
            <>
              <h4 className="run-detail__config-heading">Fitted steps</h4>
              <ol className="run-detail__steps">
                {run.pipeline_steps.map((step) => (
                  <li key={step.key}>
                    <span className="run-detail__step-label">{step.label}</span>
                    <code>{step.implementation}</code>
                    <span className="run-detail__step-detail">{step.detail}</span>
                  </li>
                ))}
              </ol>
            </>
          )}
          <LogConsole logs={run.logs || []} />
        </div>
      </section>
    </div>
  );
}
