import React from "react";
import { STATUS_META } from "../constants";
import "./StatusPill.css";

const ICONS = {
  good: "✓",
  critical: "✕",
  info: "●",
  neutral: "○",
};

/**
 * Status is never carried by colour alone — every pill pairs its tone with a
 * glyph and the status word.
 */
export default function StatusPill({ status, size = "medium", label }) {
  const meta = STATUS_META[status] || { label: status, tone: "neutral" };
  const text = label || meta.label;

  return (
    <span
      className={`status-pill status-pill--${meta.tone} status-pill--${size}`}
      title={text}
    >
      <span className="status-pill__icon" aria-hidden="true">
        {ICONS[meta.tone]}
      </span>
      {text}
    </span>
  );
}
