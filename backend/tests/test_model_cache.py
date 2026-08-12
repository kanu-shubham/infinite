"""Model cache — correctness under the conditions that make caching hard."""

import threading

import pytest

from app.ml import model_cache, registry


@pytest.fixture(autouse=True)
def clean_cache():
    model_cache.clear()
    yield
    model_cache.clear()


def _fake_model(name):
    return {"name": name}


def test_first_get_loads_and_second_hits(monkeypatch):
    calls = []
    monkeypatch.setattr(registry, "load_model", lambda rid: calls.append(rid) or _fake_model(rid))

    assert model_cache.get("run-a")["name"] == "run-a"
    assert model_cache.get("run-a")["name"] == "run-a"

    assert len(calls) == 1, "second call should not touch disk"
    stats = model_cache.stats()
    assert stats["hits"] == 1 and stats["misses"] == 1 and stats["loads"] == 1


def test_unknown_run_returns_none_and_is_not_cached(monkeypatch):
    monkeypatch.setattr(registry, "load_model", lambda rid: None)
    assert model_cache.get("nope") is None
    assert model_cache.stats()["size"] == 0


def test_lru_evicts_the_least_recently_used(monkeypatch):
    monkeypatch.setattr(registry, "load_model", lambda rid: _fake_model(rid))
    monkeypatch.setattr(model_cache, "MODEL_CACHE_SIZE", 2)

    model_cache.get("a")
    model_cache.get("b")
    model_cache.get("a")          # 'a' is now the most recent
    model_cache.get("c")          # should evict 'b'

    assert model_cache.stats()["cached_runs"] == ["a", "c"]
    assert model_cache.stats()["evictions"] == 1


def test_invalidate_forces_a_reload(monkeypatch):
    calls = []
    monkeypatch.setattr(registry, "load_model", lambda rid: calls.append(rid) or _fake_model(rid))

    model_cache.get("run-a")
    assert model_cache.invalidate("run-a") is True
    model_cache.get("run-a")

    assert len(calls) == 2


def test_concurrent_misses_load_once(monkeypatch):
    """Ten threads missing on the same id must trigger one disk read.

    Without per-key locking a cold cache under load produces a thundering herd
    of identical expensive loads — exactly when the service is least able to
    absorb it.
    """
    calls = []
    barrier = threading.Barrier(10)

    def slow_load(run_id):
        calls.append(run_id)
        return _fake_model(run_id)

    monkeypatch.setattr(registry, "load_model", slow_load)

    def worker():
        barrier.wait()            # maximise the overlap
        model_cache.get("hot")

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(calls) == 1, f"loaded {len(calls)} times, expected 1"


def test_warm_preloads(monkeypatch):
    monkeypatch.setattr(registry, "load_model", lambda rid: _fake_model(rid) if rid != "bad" else None)
    result = model_cache.warm(["a", "b", "bad"])

    assert result == {"a": True, "b": True, "bad": False}
    assert model_cache.stats()["size"] == 2
