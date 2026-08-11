/**
 * Frontend configuration for the ML pipeline feature.
 *
 * Requests go to a relative /api path by default so the CRA dev-server proxy
 * (see `proxy` in package.json) forwards them to FastAPI without CORS in the
 * loop. Point REACT_APP_ML_API_URL at a deployed backend to override.
 */
export const API_BASE_URL = process.env.REACT_APP_ML_API_URL || "";

/** How often to re-read a run while it is still queued or running. */
export const POLL_INTERVAL_MS = 1000;

export const ACTIVE_STATUSES = ["queued", "running"];

export const STATUS_META = {
  queued: { label: "Queued", tone: "neutral" },
  running: { label: "Running", tone: "info" },
  succeeded: { label: "Succeeded", tone: "good" },
  failed: { label: "Failed", tone: "critical" },
  pending: { label: "Pending", tone: "neutral" },
  skipped: { label: "Skipped", tone: "neutral" },
};

export const TABS = [
  { id: "data", label: "Data" },
  { id: "train", label: "Train" },
  { id: "runs", label: "Runs" },
];

export const TEST_SIZE_OPTIONS = [
  { value: 0.15, label: "15%" },
  { value: 0.2, label: "20%" },
  { value: 0.25, label: "25%" },
  { value: 0.3, label: "30%" },
];

export const CV_FOLD_OPTIONS = [
  { value: 0, label: "Off" },
  { value: 3, label: "3-fold" },
  { value: 5, label: "5-fold" },
];

export const DATASET_ROW_OPTIONS = [2000, 6000, 12000];

/**
 * Chart tokens. One blue hue does all the encoding work — every chart here
 * plots a single series, so a categorical palette would be colouring rank
 * rather than identity. The ramp is the sequential blue clamped at step 250
 * so even the lightest cell stays visible on white.
 */
export const CHART = {
  surface: "#ffffff",
  series: "#2a78d6",
  seriesWash: "rgba(42, 120, 214, 0.12)",
  ramp: ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"],
  grid: "#e1e0d9",
  axis: "#c3c2b7",
  muted: "#898781",
  ink: "#0b0b0b",
  inkSecondary: "#52514e",
};

/** Metrics worth a tile, in the order they should be read. */
export const METRIC_LABELS = {
  roc_auc: "ROC AUC",
  accuracy: "Accuracy",
  precision: "Precision",
  recall: "Recall",
  f1: "F1",
  r2: "R²",
  mae: "MAE",
  rmse: "RMSE",
  mape: "MAPE",
};

export const METRIC_ORDER = {
  classification: ["roc_auc", "accuracy", "precision", "recall", "f1"],
  regression: ["r2", "mae", "rmse", "mape"],
};

/** Metrics that read as percentages rather than raw units. */
export const PERCENT_METRICS = ["mape"];

/** Metrics bounded to 0–1, shown to 3 decimals. */
export const RATIO_METRICS = ["roc_auc", "accuracy", "precision", "recall", "f1", "r2"];
