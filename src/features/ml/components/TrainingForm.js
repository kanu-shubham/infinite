import React from "react";

import {
  CV_FOLD_OPTIONS,
  DATASET_ROW_OPTIONS,
  TEST_SIZE_OPTIONS,
} from "../constants";
import { formatCount } from "../utils/format";
import "./TrainingForm.css";

function Hyperparameter({ spec, value, onChange }) {
  const id = `hp-${spec.name}`;

  if (spec.kind === "choice") {
    return (
      <div className="training-form__field">
        <label className="training-form__label" htmlFor={id}>
          {spec.label}
        </label>
        <select
          id={id}
          className="training-form__control"
          value={value ?? spec.default}
          onChange={(event) => onChange(spec.name, event.target.value)}
        >
          {spec.options.map((option) => (
            <option key={option} value={option}>
              {option === "none" ? "None" : option}
            </option>
          ))}
        </select>
      </div>
    );
  }

  return (
    <div className="training-form__field">
      <label className="training-form__label" htmlFor={id}>
        {spec.label}
        <span className="training-form__range">
          {spec.minimum}–{spec.maximum}
        </span>
      </label>
      <input
        id={id}
        type="number"
        className="training-form__control"
        value={value ?? spec.default}
        min={spec.minimum}
        max={spec.maximum}
        step={spec.step}
        onChange={(event) => {
          const raw = event.target.value;
          onChange(spec.name, raw === "" ? "" : Number(raw));
        }}
      />
    </div>
  );
}

/**
 * The training form is generated from the backend catalog — targets, the
 * estimators valid for each one, and the hyperparameters each estimator
 * exposes. Nothing about the model zoo is duplicated on this side.
 */
export default function TrainingForm({
  config,
  targets,
  models,
  selectedModel,
  isReady,
  isSubmitting,
  onSelectTarget,
  onSelectModel,
  onSetHyperparameter,
  onSetField,
  onResetHyperparameters,
  onSubmit,
}) {
  const handleSubmit = (event) => {
    event.preventDefault();
    onSubmit();
  };

  return (
    <form className="training-form" onSubmit={handleSubmit}>
      <fieldset className="training-form__group">
        <legend className="training-form__legend">1 · What to predict</legend>
        <div className="training-form__targets">
          {targets.map((target) => (
            <label
              key={target.key}
              className={`target-card${
                config.target === target.key ? " target-card--active" : ""
              }`}
            >
              <input
                type="radio"
                name="target"
                value={target.key}
                checked={config.target === target.key}
                onChange={() => onSelectTarget(target.key)}
                className="target-card__input"
              />
              <span className="target-card__label">{target.label}</span>
              <span className="target-card__task">{target.task}</span>
              <span className="target-card__description">{target.description}</span>
            </label>
          ))}
        </div>
      </fieldset>

      <fieldset className="training-form__group">
        <legend className="training-form__legend">2 · Estimator</legend>
        <div className="training-form__field">
          <label className="training-form__label" htmlFor="model">
            Algorithm
          </label>
          <select
            id="model"
            className="training-form__control"
            value={config.model}
            onChange={(event) => onSelectModel(event.target.value)}
          >
            {models.map((model) => (
              <option key={model.key} value={model.key}>
                {model.label}
              </option>
            ))}
          </select>
          {selectedModel && (
            <p className="training-form__hint">{selectedModel.description}</p>
          )}
        </div>

        {selectedModel && selectedModel.params.length > 0 && (
          <>
            <div className="training-form__subhead">
              <span>Hyperparameters</span>
              <button
                type="button"
                className="training-form__link"
                onClick={onResetHyperparameters}
              >
                Reset to defaults
              </button>
            </div>
            <div className="training-form__grid">
              {selectedModel.params.map((param) => (
                <Hyperparameter
                  key={param.name}
                  spec={param}
                  value={config.hyperparameters[param.name]}
                  onChange={onSetHyperparameter}
                />
              ))}
            </div>
          </>
        )}
      </fieldset>

      <fieldset className="training-form__group">
        <legend className="training-form__legend">3 · Evaluation</legend>
        <div className="training-form__grid">
          <div className="training-form__field">
            <label className="training-form__label" htmlFor="test-size">
              Hold-out size
            </label>
            <select
              id="test-size"
              className="training-form__control"
              value={config.testSize}
              onChange={(event) => onSetField("testSize", Number(event.target.value))}
            >
              {TEST_SIZE_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          <div className="training-form__field">
            <label className="training-form__label" htmlFor="cv-folds">
              Cross-validation
            </label>
            <select
              id="cv-folds"
              className="training-form__control"
              value={config.cvFolds}
              onChange={(event) => onSetField("cvFolds", Number(event.target.value))}
            >
              {CV_FOLD_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          <div className="training-form__field">
            <label className="training-form__label" htmlFor="dataset-rows">
              Dataset rows
            </label>
            <select
              id="dataset-rows"
              className="training-form__control"
              value={config.datasetRows}
              onChange={(event) => onSetField("datasetRows", Number(event.target.value))}
            >
              {DATASET_ROW_OPTIONS.map((rows) => (
                <option key={rows} value={rows}>
                  {formatCount(rows)}
                </option>
              ))}
            </select>
          </div>

          <div className="training-form__field">
            <label className="training-form__label" htmlFor="seed">
              Random seed
            </label>
            <input
              id="seed"
              type="number"
              className="training-form__control"
              value={config.seed}
              min={0}
              onChange={(event) => onSetField("seed", Number(event.target.value))}
            />
          </div>
        </div>

        <div className="training-form__field">
          <label className="training-form__label" htmlFor="run-name">
            Run name <span className="training-form__range">optional</span>
          </label>
          <input
            id="run-name"
            type="text"
            className="training-form__control"
            placeholder="Auto-named from the estimator and target"
            maxLength={120}
            value={config.name}
            onChange={(event) => onSetField("name", event.target.value)}
          />
        </div>
      </fieldset>

      <div className="training-form__actions">
        <button
          type="submit"
          className="training-form__submit"
          disabled={!isReady || isSubmitting}
        >
          {isSubmitting ? "Starting…" : "Run pipeline"}
        </button>
        <p className="training-form__note">
          Cross-validation refits the whole pipeline once per fold, so it adds
          real time to a run.
        </p>
      </div>
    </form>
  );
}
