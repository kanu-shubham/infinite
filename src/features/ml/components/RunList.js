import React from "react";

import StatusPill from "./StatusPill";
import { formatDuration, formatMetric, formatRelative } from "../utils/format";
import "./RunList.css";

/**
 * Run history. Each entry carries enough to compare runs at a glance — the
 * estimator, the headline metric and how long it took — without opening one.
 */
export default function RunList({ runs, selectedRunId, onSelect, onDelete }) {
  if (!runs.length) {
    return (
      <p className="run-list__empty">
        No runs yet. Configure one on the Train tab and it will appear here.
      </p>
    );
  }

  return (
    <ul className="run-list">
      {runs.map((run) => {
        const isSelected = run.run_id === selectedRunId;
        const headline = run.headline_metric;

        return (
          <li key={run.run_id}>
            <div className={`run-card${isSelected ? " run-card--active" : ""}`}>
              <button
                type="button"
                className="run-card__main"
                onClick={() => onSelect(run.run_id)}
                aria-current={isSelected}
              >
                <span className="run-card__head">
                  <span className="run-card__name">{run.name}</span>
                  <StatusPill status={run.status} size="small" />
                </span>

                <span className="run-card__meta">
                  <span>{run.config.model_label}</span>
                  <span aria-hidden="true">·</span>
                  <span>{run.config.task}</span>
                  <span aria-hidden="true">·</span>
                  <span>{formatRelative(run.created_at)}</span>
                </span>

                {run.status === "succeeded" && headline && (
                  <span className="run-card__metric">
                    <span className="run-card__metric-label">{headline.label}</span>
                    <span className="run-card__metric-value">
                      {formatMetric(headline.key, headline.value)}
                    </span>
                    <span className="run-card__duration">
                      {formatDuration(run.duration_ms)}
                    </span>
                  </span>
                )}

                {run.status === "failed" && (
                  <span className="run-card__error">{run.error}</span>
                )}
              </button>

              <button
                type="button"
                className="run-card__delete"
                onClick={() => onDelete(run.run_id)}
                aria-label={`Delete run ${run.name}`}
                title="Delete run"
              >
                ✕
              </button>
            </div>
          </li>
        );
      })}
    </ul>
  );
}
