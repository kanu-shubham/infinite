"""Test the FastAPI wiring with a mocked ModelService — no model is loaded."""
from __future__ import annotations

from fastapi.testclient import TestClient

from lora_finetune.serving import api as api_module
from lora_finetune.serving.schemas import GenerateResponse, GenerateUsage


class FakeService:
    def __init__(self, *_, **__) -> None:
        self.model_path = "fake-model"
        self.device = "cpu"
        self.model = object()  # non-None => health "ok"

    def load(self) -> None:
        return None

    def generate(self, req):
        return GenerateResponse(
            text="hi there",
            finish_reason="stop",
            usage=GenerateUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
            model=self.model_path,
            latency_ms=1.0,
        )


def _client(monkeypatch, token: str | None = None):
    monkeypatch.setattr(api_module, "ModelService", FakeService)
    if token:
        monkeypatch.setenv("API_AUTH_TOKEN", token)
    else:
        monkeypatch.delenv("API_AUTH_TOKEN", raising=False)
    app = api_module.create_app("fake-model")
    return TestClient(app)


def test_health_ok(monkeypatch) -> None:
    with _client(monkeypatch) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_generate_no_auth(monkeypatch) -> None:
    with _client(monkeypatch) as client:
        r = client.post(
            "/v1/generate",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["text"] == "hi there"
        assert body["usage"]["total_tokens"] == 5


def test_generate_requires_token_when_set(monkeypatch) -> None:
    with _client(monkeypatch, token="secret") as client:
        r = client.post(
            "/v1/generate",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
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
