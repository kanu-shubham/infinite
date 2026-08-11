import React from "react";

import { formatDuration } from "../utils/format";
import "./PipelineDiagram.css";

const STAGE_ICONS = {
  succeeded: "✓",
  running: "●",
  failed: "✕",
  skipped: "–",
  pending: "",
};

/**
 * The live view of a run: one node per pipeline stage, each carrying its own
 * status, elapsed time and a one-line note from the backend. Status is shown
 * as glyph + word + colour, never colour alone.
 */
export default function PipelineDiagram({ stages, progress = 0, compact = false }) {
  const completed = stages.filter((stage) =>
    ["succeeded", "skipped"].includes(stage.status)
  ).length;

  return (
    <div className={`pipeline${compact ? " pipeline--compact" : ""}`}>
      <div className="pipeline__progress">
        <div className="pipeline__progress-head">
          <span className="pipeline__progress-label">
            {completed} of {stages.length} stages complete
          </span>
          <span className="pipeline__progress-value">{Math.round(progress * 100)}%</span>
        </div>
        <div
          className="pipeline__progress-track"
          role="progressbar"
          aria-valuenow={Math.round(progress * 100)}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label="Pipeline progress"
        >
          <div className="pipeline__progress-fill" style={{ width: `${progress * 100}%` }} />
        </div>
      </div>

      <ol className="pipeline__stages">
        {stages.map((stage, index) => (
          <li
            key={stage.key}
            className={`pipeline__stage pipeline__stage--${stage.status}`}
          >
            <div className="pipeline__stage-marker" aria-hidden="true">
              <span className="pipeline__stage-icon">
                {STAGE_ICONS[stage.status] || index + 1}
              </span>
              {index < stages.length - 1 && <span className="pipeline__connector" />}
            </div>

            <div className="pipeline__stage-body">
              <div className="pipeline__stage-head">
                <span className="pipeline__stage-label">{stage.label}</span>
                <span className="pipeline__stage-duration">
                  {stage.status === "running" ? "running…" : formatDuration(stage.duration_ms)}
                </span>
              </div>
              <p className="pipeline__stage-note">{stage.note || stage.detail}</p>
              <span className="pipeline__stage-status">{stage.status}</span>
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}
