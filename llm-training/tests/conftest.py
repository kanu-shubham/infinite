"""Shared pytest fixtures.

Most tests run on CPU without requiring any model download. Where a tiny
HF model is useful we use ``hf-internal-testing/tiny-random-gpt2`` (kept
in the HF cache if available) via a fixture that skips on offline CI.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def examples_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "examples"


@pytest.fixture(scope="session")
def configs_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "configs"


@pytest.fixture(autouse=True)
def _quiet_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")


@pytest.fixture
def offline_ok() -> bool:
    """True when tests are allowed to hit the Hugging Face Hub."""
    return os.environ.get("HF_HUB_DISABLE_TELEMETRY") != "1" or bool(os.environ.get("HF_TOKEN"))
