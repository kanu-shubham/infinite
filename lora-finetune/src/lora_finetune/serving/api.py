"""FastAPI inference service for merged or adapter-loaded models.

Features:
  * Bearer-token auth via ``API_AUTH_TOKEN``.
  * Async semaphore for bounded concurrent generation.
  * Prometheus metrics at ``/metrics``.
  * Health check at ``/health``.
  * Chat-style generation endpoint at ``/v1/generate``.
"""
from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import Counter, Histogram, make_asgi_app
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_finetune.logging_utils import get_logger, setup_logging
from lora_finetune.serving.schemas import (
    ChatMessage,
    GenerateRequest,
    GenerateResponse,
    GenerateUsage,
    HealthResponse,
)

logger = get_logger(__name__)

REQUESTS = Counter("lf_requests_total", "Requests", ["endpoint", "status"])
LATENCY = Histogram("lf_request_seconds", "Request latency", ["endpoint"])
TOKENS = Counter("lf_tokens_total", "Generated tokens", ["kind"])


def _apply_chat_template(tokenizer: Any, messages: list[ChatMessage]) -> str:
    dicts = [m.model_dump() for m in messages]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(dicts, tokenize=False, add_generation_prompt=True)
    out = []
    for m in dicts:
        out.append(f"{m['role'].upper()}: {m['content']}")
    out.append("ASSISTANT:")
    return "\n".join(out)


class ModelService:
    def __init__(self, model_path: str, adapter_path: str | None = None) -> None:
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.tokenizer: Any = None
        self.model: Any = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        logger.info("Loading model", extra={"path": self.model_path, "device": self.device})
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
        self.model.eval()

    @torch.no_grad()
    def generate(self, request: GenerateRequest) -> GenerateResponse:
        start = time.perf_counter()
        prompt = _apply_chat_template(self.tokenizer, request.messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])

        output = self.model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        completion_ids = output[0][prompt_tokens:]
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        completion_tokens = int(completion_ids.shape[0])

        if request.stop:
            for s in request.stop:
                idx = text.find(s)
                if idx >= 0:
                    text = text[:idx]
                    break

        finish = "length" if completion_tokens >= request.max_new_tokens else "stop"
        latency_ms = (time.perf_counter() - start) * 1000
        TOKENS.labels("prompt").inc(prompt_tokens)
        TOKENS.labels("completion").inc(completion_tokens)
        return GenerateResponse(
            text=text,
            finish_reason=finish,
            usage=GenerateUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            model=self.model_path,
            latency_ms=latency_ms,
        )


def create_app(
    model_path: str,
    adapter_path: str | None = None,
    max_concurrency: int = 8,
) -> FastAPI:
    setup_logging()
    service = ModelService(model_path, adapter_path)
    semaphore = asyncio.Semaphore(max_concurrency)
    auth_token = os.environ.get("API_AUTH_TOKEN")
    bearer = HTTPBearer(auto_error=False)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        service.load()
        yield

    app = FastAPI(title="lora-finetune inference", version="0.1.0", lifespan=lifespan)
    app.mount("/metrics", make_asgi_app())

    def require_auth(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> None:
        if not auth_token:
            return
        if not creds or creds.credentials != auth_token:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing token")

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        loaded = service.model is not None
        return HealthResponse(
            status="ok" if loaded else "degraded",
            model=service.model_path,
            device=service.device,
        )

    @app.post("/v1/generate", response_model=GenerateResponse, dependencies=[Depends(require_auth)])
    async def generate(req: GenerateRequest, request: Request) -> GenerateResponse:
        if semaphore.locked():
            REQUESTS.labels("generate", "throttled").inc()
        with LATENCY.labels("generate").time():
            async with semaphore:
                try:
                    resp = await asyncio.to_thread(service.generate, req)
                    REQUESTS.labels("generate", "ok").inc()
                    return resp
                except Exception as e:
                    REQUESTS.labels("generate", "error").inc()
                    logger.exception("Generation failed")
                    raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(e)) from e

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception):  # pragma: no cover
        logger.exception("Unhandled error")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc))

    return app
