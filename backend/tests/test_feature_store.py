"""Feature store behaviour — the four questions that make it a store, not a cache."""

import time

import pytest

from app.ml import feature_store
from app.ml.dataset import feature_columns


@pytest.fixture(autouse=True)
def clean_store():
    feature_store.reset_backend_for_tests()
    yield
    feature_store.reset_backend_for_tests()


def test_materialize_writes_every_entity():
    result = feature_store.materialize("cancellation", limit=50)
    assert result["entities"] == 50
    assert result["features_per_entity"] == 20


def test_lookup_returns_the_features_that_were_written():
    feature_store.materialize("cancellation", limit=10)
    [found] = feature_store.lookup("cancellation", ["BK-000001"])

    assert found.found is True
    assert found.entity_id == "BK-000001"
    columns = feature_columns("cancellation")
    assert set(found.features) == set(columns["numeric"] + columns["categorical"])


def test_lookup_reports_a_miss_rather_than_raising():
    feature_store.materialize("cancellation", limit=5)
    [missing] = feature_store.lookup("cancellation", ["BK-999999"])

    assert missing.found is False
    assert missing.features == {}
    assert missing.staleness_seconds is None


def test_batch_lookup_preserves_request_order():
    feature_store.materialize("cancellation", limit=20)
    ids = ["BK-000003", "BK-999999", "BK-000001"]
    results = feature_store.lookup("cancellation", ids)

    assert [r.entity_id for r in results] == ids
    assert [r.found for r in results] == [True, False, True]


def test_staleness_is_reported_and_grows():
    feature_store.materialize("cancellation", limit=5)
    first = feature_store.lookup("cancellation", ["BK-000001"])[0].staleness_seconds
    time.sleep(0.05)
    second = feature_store.lookup("cancellation", ["BK-000001"])[0].staleness_seconds

    assert first is not None and second > first


def test_schema_mismatch_fails_loudly(monkeypatch):
    """The parity check: a store written for different features must not serve.

    This is the whole point of the fingerprint — online/offline skew otherwise
    produces a silently worse model rather than an error.
    """
    feature_store.materialize("cancellation", limit=5)
    monkeypatch.setattr(feature_store, "schema_hash", lambda names: "different-hash")

    with pytest.raises(feature_store.FeatureStoreError, match="schema mismatch"):
        feature_store.lookup("cancellation", ["BK-000001"])


def test_max_staleness_rejects_old_rows(monkeypatch):
    feature_store.materialize("cancellation", limit=5)
    monkeypatch.setattr(feature_store, "FEATURE_MAX_STALENESS_SECONDS", 1)
    time.sleep(1.1)

    with pytest.raises(feature_store.FeatureStoreError, match="old"):
        feature_store.lookup("cancellation", ["BK-000001"])


def test_ttl_expiry_makes_a_row_a_miss():
    feature_store.materialize("cancellation", limit=3, ttl=1)
    assert feature_store.lookup("cancellation", ["BK-000001"])[0].found is True
    time.sleep(1.2)
    assert feature_store.lookup("cancellation", ["BK-000001"])[0].found is False


def test_to_frame_gives_one_row_per_request_even_for_misses():
    """A miss becomes an all-NaN row, not a dropped row.

    Dropping would silently misalign predictions with the ids that asked for
    them — the caller would get 2 answers for 3 questions and no indication
    which was which.
    """
    feature_store.materialize("cancellation", limit=5)
    results = feature_store.lookup("cancellation", ["BK-000001", "BK-999999"])
    frame = feature_store.to_frame("cancellation", results)

    columns = feature_columns("cancellation")
    assert len(frame) == 2
    assert list(frame.columns) == columns["numeric"] + columns["categorical"]
    assert frame.iloc[1].isna().all()


def test_price_target_frame_excludes_the_target_column():
    feature_store.materialize("price", limit=5)
    results = feature_store.lookup("price", ["BK-000001"])
    frame = feature_store.to_frame("price", results)

    assert "adr" not in frame.columns
    assert frame.shape[1] == 19


def test_stats_reports_backend_and_population():
    feature_store.materialize("cancellation", limit=25)
    stats = feature_store.stats("cancellation")

    assert stats["entities"] == 25
    assert stats["backend"] in {"redis", "memory"}
    assert len(stats["expected_schema_hash"]) == 16
