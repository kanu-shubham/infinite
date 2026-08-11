import { useCallback } from "react";

import { fetchCatalog, fetchDataset } from "../services/mlApi";
import useAsyncResource from "./useAsyncResource";

/** Targets, estimators and their tunable hyperparameters. */
export function useCatalog() {
  const loader = useCallback(() => fetchCatalog(), []);
  const { data, isLoading, error, reload } = useAsyncResource(loader);
  return { catalog: data, isLoading, error, reload };
}

/** Profile, feature specs and a preview slice for one target. */
export function useDataset(target) {
  const loader = useCallback(() => fetchDataset(target), [target]);
  const { data, isLoading, error, reload } = useAsyncResource(loader, {
    enabled: Boolean(target),
  });
  return { dataset: data, isLoading, error, reload };
}
