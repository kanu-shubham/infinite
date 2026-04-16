"""
materialize.py
--------------
Loads features from the offline store (CSV) into the online store (SQLite/Redis).

WHY materialize?
  The offline store holds all historical data – great for training, slow for
  serving (can't query BigQuery in 5 ms during inference!).
  Materialization copies the LATEST feature values to the online store so
  your prediction API can retrieve them in < 10 ms.

Run this on a schedule (e.g., hourly Airflow task) to keep the online store fresh.

Usage:
    python materialize.py
"""

from datetime import datetime, timezone
from pathlib import Path

from feast import FeatureStore


REPO_PATH = Path(__file__).parent / "feature_repo"


def materialize():
    store = FeatureStore(repo_path=str(REPO_PATH))

    end_date   = datetime.now(tz=timezone.utc)

    print("Materializing features to online store …")
    # materialize_incremental only updates rows newer than the last run,
    # which is much faster than a full reload.
    store.materialize_incremental(end_date=end_date)
    print("Done. Online store is now up-to-date.")


if __name__ == "__main__":
    materialize()
