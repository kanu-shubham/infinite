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
from app.ml import feature_store, model_cache, registry  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A TestClient with an isolated registry, model cache and feature store.

    All three are module-level singletons, so leaving any of them dirty leaks
    across tests. The model cache is the sharp one: run ids are timestamps at
    second resolution, so two tests running inside the same second generate the
    *same* id — and a cache keyed on that id will serve the first test's model
    to the second.
    """
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    monkeypatch.setattr(registry, "RUNS_DIR", runs_dir)
    registry.reset_for_tests()
    model_cache.clear()
    feature_store.reset_backend_for_tests()

    with TestClient(create_app()) as test_client:
        yield test_client

    registry.reset_for_tests()
    model_cache.clear()
    feature_store.reset_backend_for_tests()
