"""HTTP surface for the training and serving pipeline."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from ..ml import inference, registry, training
from ..ml.dataset import (
    DEFAULT_TARGET,
    TARGETS,
    dataset_profile,
    feature_specs,
    sample_rows,
)
from ..ml.models import DEFAULT_MODEL_FOR_TASK, models_for_task
from ..ml.pipeline import PIPELINE_STEPS
from ..ml.training import STAGES
from ..schemas import (
    DeleteResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    RunAccepted,
    TrainRequest,
)

router = APIRouter(prefix="/api")

API_VERSION = "1.0.0"


def _require_run(run_id: str) -> Dict[str, Any]:
    run = registry.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return run


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok", version=API_VERSION, runs=len(registry.list_summaries(limit=1000))
    )


@router.get("/catalog")
def catalog() -> Dict[str, Any]:
    """Targets, estimators and stage definitions — everything the form needs."""
    return {
        "targets": [
            {
                "key": key,
                "task": spec["task"],
                "label": spec["label"],
                "description": spec["description"],
                "default_model": DEFAULT_MODEL_FOR_TASK[spec["task"]],
                "models": models_for_task(spec["task"]),
            }
            for key, spec in TARGETS.items()
        ],
        "stages": STAGES,
        "pipeline_steps": PIPELINE_STEPS,
        "defaults": {"target": DEFAULT_TARGET, "test_size": 0.2, "cv_folds": 0},
    }


@router.get("/dataset")
def dataset(target: str = Query(default=DEFAULT_TARGET)) -> Dict[str, Any]:
    if target not in TARGETS:
        raise HTTPException(status_code=422, detail=f"Unknown target '{target}'.")
    return {
        "profile": dataset_profile(target),
        "features": [spec.__dict__ for spec in feature_specs(target)],
        "preview": sample_rows(5, seed=7),
    }


@router.get("/dataset/sample")
def dataset_sample(
    target: str = Query(default=DEFAULT_TARGET),
    count: int = Query(default=1, ge=1, le=20),
) -> Dict[str, Any]:
    """Random real rows, used to prefill the prediction form."""
    if target not in TARGETS:
        raise HTTPException(status_code=422, detail=f"Unknown target '{target}'.")
    return {"rows": sample_rows(count)}


@router.post("/runs", response_model=RunAccepted, status_code=202)
def create_run(request: TrainRequest, background: BackgroundTasks) -> RunAccepted:
    """Queue a training run and return immediately; the client polls for state."""
    record = training.build_run_record(request.to_config(), request.name)
    registry.create(record)
    background.add_task(training.execute_run, record["run_id"])
    return RunAccepted(run_id=record["run_id"], status=record["status"], name=record["name"])


@router.get("/runs")
def list_runs(
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=25, ge=1, le=100),
) -> Dict[str, List[Dict[str, Any]]]:
    return {"runs": registry.list_summaries(status=status, limit=limit)}


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> Dict[str, Any]:
    return _require_run(run_id)


@router.delete("/runs/{run_id}", response_model=DeleteResponse)
def delete_run(run_id: str) -> DeleteResponse:
    if not registry.delete(run_id):
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return DeleteResponse(run_id=run_id, deleted=True)


@router.post("/runs/{run_id}/predict", response_model=PredictResponse)
def predict(run_id: str, request: PredictRequest) -> PredictResponse:
    run = _require_run(run_id)
    try:
        predictions = inference.predict(run, request.rows)
    except inference.ModelUnavailable as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=f"Could not score the payload: {error}") from error

    return PredictResponse(
        run_id=run_id,
        task=run["config"]["task"],
        target=run["config"]["target"],
        predictions=predictions,
    )
