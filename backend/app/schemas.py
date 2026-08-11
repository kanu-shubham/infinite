"""Request and response models for the API layer."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .config import DATASET_ROWS, DATASET_SEED
from .ml.dataset import DEFAULT_TARGET, TARGETS
from .ml.models import DEFAULT_MODEL_FOR_TASK, MODEL_CATALOG


class TrainRequest(BaseModel):
    """Everything needed to launch one training run."""

    target: str = Field(default=DEFAULT_TARGET, description="Prediction target key")
    model: Optional[str] = Field(default=None, description="Estimator key from the catalog")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    cv_folds: int = Field(default=0, ge=0, le=10)
    dataset_rows: int = Field(default=DATASET_ROWS, ge=500, le=20000)
    dataset_seed: int = Field(default=DATASET_SEED, ge=0, le=10_000_000)
    seed: int = Field(default=42, ge=0, le=10_000_000)
    name: Optional[str] = Field(default=None, max_length=120)

    @field_validator("target")
    @classmethod
    def _known_target(cls, value: str) -> str:
        if value not in TARGETS:
            raise ValueError(f"Unknown target '{value}'. Expected one of {sorted(TARGETS)}.")
        return value

    @field_validator("cv_folds")
    @classmethod
    def _usable_fold_count(cls, value: int) -> int:
        if value == 1:
            raise ValueError("cv_folds must be 0 (disabled) or at least 2.")
        return value

    @model_validator(mode="after")
    def _model_matches_task(self) -> "TrainRequest":
        task = TARGETS[self.target]["task"]

        if self.model is None:
            self.model = DEFAULT_MODEL_FOR_TASK[task]
            return self

        spec = MODEL_CATALOG.get(self.model)
        if spec is None:
            raise ValueError(f"Unknown model '{self.model}'.")
        if spec["task"] != task:
            raise ValueError(
                f"Model '{self.model}' is a {spec['task']} estimator but "
                f"target '{self.target}' is a {task} problem."
            )
        return self

    def to_config(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "task": TARGETS[self.target]["task"],
            "model": self.model,
            "model_label": MODEL_CATALOG[self.model]["label"],
            "hyperparameters": self.hyperparameters,
            "test_size": self.test_size,
            "cv_folds": self.cv_folds,
            "dataset_rows": self.dataset_rows,
            "dataset_seed": self.dataset_seed,
            "seed": self.seed,
        }


class PredictRequest(BaseModel):
    """One or more bookings to score with a finished run's model."""

    rows: List[Dict[str, Any]] = Field(min_length=1, max_length=100)


class Prediction(BaseModel):
    prediction: float
    label: Optional[str] = None
    probability: Optional[float] = None


class PredictResponse(BaseModel):
    run_id: str
    task: str
    target: str
    predictions: List[Prediction]


class RunAccepted(BaseModel):
    run_id: str
    status: str
    name: str


class DeleteResponse(BaseModel):
    run_id: str
    deleted: bool


class HealthResponse(BaseModel):
    status: str
    version: str
    runs: int
