import React, { useCallback, useEffect, useMemo, useState } from "react";

import DatasetPanel from "./components/DatasetPanel";
import ErrorMessage from "../../components/common/ErrorMessage";
import LoadingSpinner from "../../components/common/LoadingSpinner";
import PipelineDiagram from "./components/PipelineDiagram";
import PredictionPanel from "./components/PredictionPanel";
import RunDetail from "./components/RunDetail";
import RunList from "./components/RunList";
import StatusPill from "./components/StatusPill";
import TrainingForm from "./components/TrainingForm";
import { TABS } from "./constants";
import { useCatalog, useDataset } from "./hooks/useCatalog";
import useRunDetail from "./hooks/useRunDetail";
import useRuns from "./hooks/useRuns";
import useTrainingConfig from "./hooks/useTrainingConfig";
import "./MlPipelinePage.css";

/**
 * The ML pipeline workspace.
 *
 * Three tabs walk the same lifecycle the backend does: look at the data,
 * configure and launch a run, then inspect what it produced and score new
 * bookings against it.
 */
export default function MlPipelinePage() {
  const [activeTab, setActiveTab] = useState("data");
  const [selectedRunId, setSelectedRunId] = useState(null);
  const [launchedRunId, setLaunchedRunId] = useState(null);
  const [submission, setSubmission] = useState({ isSubmitting: false, error: null });

  const { catalog, isLoading: catalogLoading, error: catalogError, reload: reloadCatalog } =
    useCatalog();
  const training = useTrainingConfig(catalog);
  const { runs, isLoading: runsLoading, error: runsError, submit, remove } = useRuns();

  const { dataset, isLoading: datasetLoading, error: datasetError } = useDataset(
    training.config.target
  );

  // Default the Runs tab to the newest run so it is never blank.
  useEffect(() => {
    if (!selectedRunId && runs.length) setSelectedRunId(runs[0].run_id);
  }, [runs, selectedRunId]);

  const { run: selectedRun, error: runError } = useRunDetail(selectedRunId);
  const { run: launchedRun } = useRunDetail(launchedRunId);

  const sourceColumns = useMemo(() => {
    if (!dataset) return [];
    return [
      ...dataset.profile.numeric_features,
      ...dataset.profile.categorical_features,
    ];
  }, [dataset]);

  const handleSubmit = useCallback(async () => {
    setSubmission({ isSubmitting: true, error: null });
    try {
      const accepted = await submit(training.toRequest());
      setLaunchedRunId(accepted.run_id);
      setSelectedRunId(accepted.run_id);
      setSubmission({ isSubmitting: false, error: null });
    } catch (error) {
      setSubmission({ isSubmitting: false, error: error.message });
    }
  }, [submit, training]);

  const handleDelete = useCallback(
    async (runId) => {
      await remove(runId);
      setSelectedRunId((previous) => (previous === runId ? null : previous));
      setLaunchedRunId((previous) => (previous === runId ? null : previous));
    },
    [remove]
  );

  if (catalogLoading) {
    return (
      <div className="ml-page">
        <LoadingSpinner text="Connecting to the pipeline API…" />
      </div>
    );
  }

  if (catalogError) {
    return (
      <div className="ml-page">
        <ErrorMessage message={catalogError} onRetry={reloadCatalog} />
      </div>
    );
  }

  return (
    <div className="ml-page">
      <header className="ml-page__header">
        <div>
          <h1 className="ml-page__title">Hotel Bookings ML Pipeline</h1>
          <p className="ml-page__subtitle">
            A scikit-learn pipeline — feature engineering, imputation, encoding
            and a supervised estimator — trained and served from FastAPI, driven
            from this React front end.
          </p>
        </div>
        {/* The catalog only renders once the API answered, so reaching here
            means the backend is reachable. */}
        <StatusPill status="succeeded" label="Pipeline API online" />
      </header>

      <div className="ml-page__tabs" role="tablist" aria-label="Pipeline workspace">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            role="tab"
            aria-selected={activeTab === tab.id}
            className={`ml-tab${activeTab === tab.id ? " ml-tab--active" : ""}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
            {tab.id === "runs" && runs.length > 0 && (
              <span className="ml-tab__count">{runs.length}</span>
            )}
          </button>
        ))}
      </div>

      {/* ── Data ────────────────────────────────────────────────────────── */}
      {activeTab === "data" && (
        <section className="ml-page__section">
          {datasetLoading && <LoadingSpinner text="Profiling the dataset…" />}
          {datasetError && <ErrorMessage message={datasetError} />}
          {dataset && !datasetError && <DatasetPanel dataset={dataset} />}
        </section>
      )}

      {/* ── Train ───────────────────────────────────────────────────────── */}
      {activeTab === "train" && (
        <section className="ml-page__section ml-page__split">
          <div className="ml-page__panel">
            <h2 className="ml-page__panel-title">Configure a run</h2>
            {submission.error && <ErrorMessage message={submission.error} />}
            <TrainingForm
              config={training.config}
              targets={training.targets}
              models={training.models}
              selectedModel={training.selectedModel}
              isReady={training.isReady}
              isSubmitting={submission.isSubmitting}
              onSelectTarget={training.selectTarget}
              onSelectModel={training.selectModel}
              onSetHyperparameter={training.setHyperparameter}
              onSetField={training.setField}
              onResetHyperparameters={training.resetHyperparameters}
              onSubmit={handleSubmit}
            />
          </div>

          <div className="ml-page__panel ml-page__panel--sticky">
            <h2 className="ml-page__panel-title">Live run</h2>
            {launchedRun ? (
              <>
                <div className="ml-page__live-head">
                  <span className="ml-page__live-name">{launchedRun.name}</span>
                  <StatusPill status={launchedRun.status} size="small" />
                </div>
                <PipelineDiagram
                  stages={launchedRun.stages}
                  progress={launchedRun.progress}
                />
                {launchedRun.status === "succeeded" && (
                  <button
                    type="button"
                    className="ml-page__jump"
                    onClick={() => {
                      setSelectedRunId(launchedRun.run_id);
                      setActiveTab("runs");
                    }}
                  >
                    View results →
                  </button>
                )}
                {launchedRun.status === "failed" && (
                  <p className="ml-page__live-error">{launchedRun.error}</p>
                )}
              </>
            ) : (
              <p className="ml-page__placeholder">
                Launch a run and every stage — ingest, validate, split,
                preprocess, train, evaluate, register — reports here as it
                completes.
              </p>
            )}
          </div>
        </section>
      )}

      {/* ── Runs ────────────────────────────────────────────────────────── */}
      {activeTab === "runs" && (
        <section className="ml-page__section ml-page__split ml-page__split--history">
          <div className="ml-page__panel ml-page__panel--sticky">
            <h2 className="ml-page__panel-title">History</h2>
            {runsLoading && !runs.length && <LoadingSpinner size="small" text="" />}
            {runsError && <ErrorMessage message={runsError} />}
            <RunList
              runs={runs}
              selectedRunId={selectedRunId}
              onSelect={setSelectedRunId}
              onDelete={handleDelete}
            />
          </div>

          <div className="ml-page__detail">
            {runError && <ErrorMessage message={runError} />}
            {!runError && !selectedRun && (
              <p className="ml-page__placeholder">
                Select a run to see its stages, metrics and diagnostics.
              </p>
            )}
            {selectedRun && (
              <RunDetail run={selectedRun} sourceColumns={sourceColumns}>
                <PredictionPanel run={selectedRun} />
              </RunDetail>
            )}
          </div>
        </section>
      )}
    </div>
  );
}
