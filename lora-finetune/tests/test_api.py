"""Test the FastAPI wiring with a mocked ModelService — no model is loaded."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from lora_finetune.serving import api as api_module
from lora_finetune.serving.schemas import GenerateResponse, GenerateUsage


class FakeService:
    def __init__(self, model_path, adapters=None, load_in_4bit=False, **__) -> None:
        self.model_path = model_path
        self.adapter_paths = adapters or {}
        self.device = "cpu"
        self.model = object()  # non-None => health "ok"
        self.default_adapter = next(iter(self.adapter_paths), None)
        self.last_request = None

    def load(self) -> None:
        return None

    def list_adapters(self):
        from lora_finetune.serving.schemas import AdapterInfo

        return [
            AdapterInfo(name=n, path=p, is_default=(n == self.default_adapter))
            for n, p in self.adapter_paths.items()
        ]

    def generate(self, req):
        self.last_request = req
        if self.adapter_paths and req.adapter and req.adapter not in self.adapter_paths:
            raise KeyError(f"Unknown adapter: {req.adapter}")
        return GenerateResponse(
            text="hi there",
            finish_reason="stop",
            usage=GenerateUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
            model=self.model_path,
            adapter=req.adapter or self.default_adapter,
            latency_ms=1.0,
        )


def _client(monkeypatch, token: str | None = None, adapters=None):
    monkeypatch.setattr(api_module, "ModelService", FakeService)
    if token:
        monkeypatch.setenv("API_AUTH_TOKEN", token)
    else:
        monkeypatch.delenv("API_AUTH_TOKEN", raising=False)
    app = api_module.create_app("fake-model", adapters=adapters)
    return TestClient(app)


def test_health_ok(monkeypatch) -> None:
    with _client(monkeypatch) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["adapters"] == []


def test_generate_no_auth(monkeypatch) -> None:
    with _client(monkeypatch) as client:
        r = client.post("/v1/generate", json={"messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200
        body = r.json()
        assert body["text"] == "hi there"
        assert body["usage"]["total_tokens"] == 5


def test_generate_requires_token_when_set(monkeypatch) -> None:
    with _client(monkeypatch, token="secret") as client:
        r = client.post("/v1/generate", json={"messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 401

        r = client.post(
            "/v1/generate",
            headers={"Authorization": "Bearer secret"},
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200


def test_generate_rejects_empty_messages(monkeypatch) -> None:
    with _client(monkeypatch) as client:
        r = client.post("/v1/generate", json={"messages": []})
        assert r.status_code == 422


def test_multi_adapter_health_lists_adapters(monkeypatch) -> None:
    adapters = ["support=outputs/support", "sales=outputs/sales"]
    with _client(monkeypatch, adapters=adapters) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert sorted(r.json()["adapters"]) == ["sales", "support"]


def test_v1_adapters_endpoint(monkeypatch) -> None:
    adapters = ["a=p1", "b=p2"]
    with _client(monkeypatch, adapters=adapters) as client:
        r = client.get("/v1/adapters")
        assert r.status_code == 200
        names = [a["name"] for a in r.json()["adapters"]]
        assert names == ["a", "b"]
        assert r.json()["adapters"][0]["is_default"] is True


def test_unknown_adapter_returns_400(monkeypatch) -> None:
    adapters = ["a=p1"]
    with _client(monkeypatch, adapters=adapters) as client:
        r = client.post(
            "/v1/generate",
            json={"messages": [{"role": "user", "content": "hi"}], "adapter": "missing"},
        )
        assert r.status_code == 400


def test_parse_adapters_accepts_name_path() -> None:
    parsed = api_module._parse_adapters(["a=path/a", "b=path/b"])
    assert parsed == {"a": "path/a", "b": "path/b"}


def test_parse_adapters_bare_path_uses_default_name() -> None:
    parsed = api_module._parse_adapters(["only/path"])
    assert parsed == {"default": "only/path"}


def test_parse_adapters_rejects_duplicates() -> None:
    with pytest.raises(ValueError):
        api_module._parse_adapters(["a=p1", "a=p2"])
