import { API_BASE_URL } from "../constants";

/**
 * Thin fetch wrapper around the pipeline API.
 *
 * Every call funnels through `request` so error handling is uniform: FastAPI
 * reports validation problems as a `detail` field (a string for HTTPException,
 * a list of field errors for a 422), and both are flattened into one readable
 * message the UI can drop straight into <ErrorMessage />.
 */

function formatDetail(detail) {
  if (typeof detail === "string") return detail;

  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        const field = Array.isArray(item.loc) ? item.loc.slice(1).join(".") : "";
        return field ? `${field}: ${item.msg}` : item.msg;
      })
      .join("; ");
  }

  return "";
}

async function request(path, options = {}) {
  let response;

  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch (networkError) {
    throw new Error(
      "Cannot reach the pipeline API. Start it with `uvicorn app.main:app --reload` from the backend directory."
    );
  }

  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      message = formatDetail(body.detail) || message;
    } catch (parseError) {
      // A non-JSON error body — keep the status-code message.
    }
    throw new Error(message);
  }

  if (response.status === 204) return null;
  return response.json();
}

export function fetchHealth() {
  return request("/api/health");
}

export function fetchCatalog() {
  return request("/api/catalog");
}

export function fetchDataset(target) {
  return request(`/api/dataset?target=${encodeURIComponent(target)}`);
}

export function fetchSampleRows(target, count = 1) {
  return request(
    `/api/dataset/sample?target=${encodeURIComponent(target)}&count=${count}`
  );
}

export function fetchRuns({ status, limit = 25 } = {}) {
  const params = new URLSearchParams({ limit: String(limit) });
  if (status) params.set("status", status);
  return request(`/api/runs?${params.toString()}`);
}

export function fetchRun(runId) {
  return request(`/api/runs/${encodeURIComponent(runId)}`);
}

export function createRun(config) {
  return request("/api/runs", { method: "POST", body: JSON.stringify(config) });
}

export function deleteRun(runId) {
  return request(`/api/runs/${encodeURIComponent(runId)}`, { method: "DELETE" });
}

export function predict(runId, rows) {
  return request(`/api/runs/${encodeURIComponent(runId)}/predict`, {
    method: "POST",
    body: JSON.stringify({ rows }),
  });
}
