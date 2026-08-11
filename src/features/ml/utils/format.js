import { METRIC_LABELS, PERCENT_METRICS, RATIO_METRICS } from "../constants";

export function metricLabel(key) {
  return METRIC_LABELS[key] || key.replace(/_/g, " ").toUpperCase();
}

export function formatMetric(key, value) {
  if (value === null || value === undefined) return "—";
  if (PERCENT_METRICS.includes(key)) return `${value.toFixed(1)}%`;
  if (RATIO_METRICS.includes(key)) return value.toFixed(3);
  return value.toFixed(2);
}

export function formatDuration(milliseconds) {
  if (milliseconds === null || milliseconds === undefined) return "—";
  if (milliseconds < 1000) return `${Math.round(milliseconds)} ms`;
  if (milliseconds < 60000) return `${(milliseconds / 1000).toFixed(1)} s`;
  const minutes = Math.floor(milliseconds / 60000);
  const seconds = Math.round((milliseconds % 60000) / 1000);
  return `${minutes}m ${seconds}s`;
}

export function formatCount(value) {
  if (value === null || value === undefined) return "—";
  return value.toLocaleString("en-US");
}

export function formatTimestamp(iso) {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function formatRelative(iso) {
  if (!iso) return "—";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "—";

  const seconds = Math.max(0, Math.round((Date.now() - then) / 1000));
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

/** "market_segment" → "Market segment" */
export function humanize(name) {
  const spaced = name.replace(/_/g, " ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

/**
 * One-hot columns arrive as "deposit_type_Non Refund". Split the encoded
 * level off the source column so importance bars stay readable.
 */
export function splitEncodedFeature(name, sourceColumns = []) {
  const source = sourceColumns.find(
    (column) => name === column || name.startsWith(`${column}_`)
  );
  if (!source || source === name) return { base: humanize(name), level: null };
  return { base: humanize(source), level: name.slice(source.length + 1) };
}
