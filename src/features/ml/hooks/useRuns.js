import { useCallback, useMemo } from "react";

import { ACTIVE_STATUSES } from "../constants";
import { createRun, deleteRun, fetchRuns } from "../services/mlApi";
import useAsyncResource from "./useAsyncResource";
import usePolling from "./usePolling";

/**
 * The run history, refreshed on an interval for as long as any run is still
 * in flight. Once everything has settled the polling stops on its own.
 */
export default function useRuns() {
  const loader = useCallback(() => fetchRuns({ limit: 25 }), []);
  const { data, isLoading, error, reload } = useAsyncResource(loader);

  const runs = useMemo(() => data?.runs ?? [], [data]);
  const hasActiveRun = runs.some((run) => ACTIVE_STATUSES.includes(run.status));

  usePolling(reload, hasActiveRun);

  const submit = useCallback(
    async (config) => {
      const accepted = await createRun(config);
      await reload();
      return accepted;
    },
    [reload]
  );

  const remove = useCallback(
    async (runId) => {
      await deleteRun(runId);
      await reload();
    },
    [reload]
  );

  return { runs, isLoading, error, hasActiveRun, reload, submit, remove };
}
