import { useCallback } from "react";

import { ACTIVE_STATUSES } from "../constants";
import { fetchRun } from "../services/mlApi";
import useAsyncResource from "./useAsyncResource";
import usePolling from "./usePolling";

/**
 * Full detail for one run — stages, metrics, importances, logs — polled while
 * the run is still executing so the pipeline diagram animates in real time.
 */
export default function useRunDetail(runId) {
  const loader = useCallback(() => fetchRun(runId), [runId]);
  const { data, isLoading, error, reload } = useAsyncResource(loader, {
    enabled: Boolean(runId),
  });

  const isActive = Boolean(data) && ACTIVE_STATUSES.includes(data.status);
  usePolling(reload, isActive);

  return { run: runId ? data : null, isLoading: Boolean(runId) && isLoading, error, reload };
}
