import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

# Point artifacts at a scratch dir and shrink the dataset before the app
# modules read their config at import time.
os.environ.setdefault("ML_ARTIFACTS_DIR", str(BACKEND_ROOT / ".pytest-artifacts"))
os.environ.setdefault("ML_DATASET_ROWS", "1200")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app  # noqa: E402
from app.ml import registry  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A TestClient with an isolated, empty run registry."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    monkeypatch.setattr(registry, "RUNS_DIR", runs_dir)
    registry.reset_for_tests()

    with TestClient(create_app()) as test_client:
        yield test_client

    registry.reset_for_tests()
