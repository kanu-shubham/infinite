import { useCallback, useEffect, useMemo, useState } from "react";

import { DATASET_ROW_OPTIONS } from "../constants";

const INITIAL = {
  target: "",
  model: "",
  hyperparameters: {},
  testSize: 0.2,
  cvFolds: 0,
  datasetRows: DATASET_ROW_OPTIONS[1],
  seed: 42,
  name: "",
};

function defaultsFor(modelSpec) {
  if (!modelSpec) return {};
  return modelSpec.params.reduce(
    (accumulator, param) => ({ ...accumulator, [param.name]: param.default }),
    {}
  );
}

/**
 * Owns the training form.
 *
 * The catalog is the single source of truth for which estimators pair with
 * which target and what each one exposes, so switching target re-seeds the
 * model and its hyperparameters instead of carrying stale values across.
 */
export default function useTrainingConfig(catalog) {
  const [config, setConfig] = useState(INITIAL);

  const targets = useMemo(() => catalog?.targets ?? [], [catalog]);

  const selectedTarget = useMemo(
    () => targets.find((target) => target.key === config.target) ?? null,
    [targets, config.target]
  );

  const models = useMemo(() => selectedTarget?.models ?? [], [selectedTarget]);

  const selectedModel = useMemo(
    () => models.find((model) => model.key === config.model) ?? null,
    [models, config.model]
  );

  // Seed the form once the catalog lands.
  useEffect(() => {
    if (!catalog || config.target) return;
    const first = catalog.targets.find((t) => t.key === catalog.defaults.target) ?? catalog.targets[0];
    if (!first) return;
    const model = first.models.find((m) => m.key === first.default_model) ?? first.models[0];
    setConfig((previous) => ({
      ...previous,
      target: first.key,
      model: model?.key ?? "",
      hyperparameters: defaultsFor(model),
      testSize: catalog.defaults.test_size,
      cvFolds: catalog.defaults.cv_folds,
    }));
  }, [catalog, config.target]);

  const selectTarget = useCallback(
    (targetKey) => {
      const target = targets.find((item) => item.key === targetKey);
      if (!target) return;
      const model =
        target.models.find((item) => item.key === target.default_model) ?? target.models[0];
      setConfig((previous) => ({
        ...previous,
        target: targetKey,
        model: model?.key ?? "",
        hyperparameters: defaultsFor(model),
      }));
    },
    [targets]
  );

  const selectModel = useCallback(
    (modelKey) => {
      const model = models.find((item) => item.key === modelKey);
      setConfig((previous) => ({
        ...previous,
        model: modelKey,
        hyperparameters: defaultsFor(model),
      }));
    },
    [models]
  );

  const setHyperparameter = useCallback((name, value) => {
    setConfig((previous) => ({
      ...previous,
      hyperparameters: { ...previous.hyperparameters, [name]: value },
    }));
  }, []);

  const setField = useCallback((field, value) => {
    setConfig((previous) => ({ ...previous, [field]: value }));
  }, []);

  const resetHyperparameters = useCallback(() => {
    setConfig((previous) => ({ ...previous, hyperparameters: defaultsFor(selectedModel) }));
  }, [selectedModel]);

  const toRequest = useCallback(
    () => ({
      target: config.target,
      model: config.model,
      hyperparameters: config.hyperparameters,
      test_size: config.testSize,
      cv_folds: config.cvFolds,
      dataset_rows: config.datasetRows,
      seed: config.seed,
      name: config.name.trim() || null,
    }),
    [config]
  );

  return {
    config,
    targets,
    models,
    selectedTarget,
    selectedModel,
    isReady: Boolean(config.target && config.model),
    selectTarget,
    selectModel,
    setHyperparameter,
    setField,
    resetHyperparameters,
    toRequest,
  };
}
