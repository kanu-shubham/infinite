"""FastAPI inference service.

Supports three deployment shapes:

  1. **Merged model** — pass ``--model`` only. Plain HF generate, no PEFT runtime.
  2. **Single adapter on base** — pass ``--model BASE --adapter PATH`` to keep
     the adapter mounted dynamically.
  3. **Multi-adapter** — pass ``--adapter NAME=PATH`` repeatedly. Clients pick
     an adapter per request via the ``adapter`` field. One pod, many tenants.

Quantized base models load transparently:

  * **GPTQ / AWQ** — pass the on-disk quant directory as ``--model``; transformers
    reads the embedded quant config.
  * **bitsandbytes 4-bit** at serve time — set ``LF_LOAD_IN_4BIT=1``.

Other features:

  * Bearer-token auth via ``API_AUTH_TOKEN``.
  * Async semaphore for bounded concurrent generation.
  * Per-adapter Prometheus metrics at ``/metrics``.
  * Health/info endpoints at ``/health`` and ``/v1/adapters``.
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import Counter, Histogram, make_asgi_app
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from lora_finetune.logging_utils import get_logger, setup_logging
from lora_finetune.serving.schemas import (
    AdapterInfo,
    AdaptersResponse,
    ChatMessage,
    GenerateRequest,
    GenerateResponse,
    GenerateUsage,
    HealthResponse,
)

logger = get_logger(__name__)

REQUESTS = Counter("lf_requests_total", "Requests", ["endpoint", "adapter", "status"])
LATENCY = Histogram("lf_request_seconds", "Request latency", ["endpoint", "adapter"])
TOKENS = Counter("lf_tokens_total", "Generated tokens", ["adapter", "kind"])


def _apply_chat_template(tokenizer: Any, messages: list[ChatMessage]) -> str:
    dicts = [m.model_dump() for m in messages]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(dicts, tokenize=False, add_generation_prompt=True)
    out = [f"{m['role'].upper()}: {m['content']}" for m in dicts]
    out.append("ASSISTANT:")
    return "\n".join(out)


def _parse_adapters(items: list[str] | None) -> dict[str, str]:
    """Parse ``--adapter`` strings into a ``{name: path}`` mapping.

    Accepts ``name=path`` or a bare ``path`` (uses ``"default"`` as the name).
    """
    out: dict[str, str] = {}
    for raw in items or []:
        if "=" in raw:
            name, _, path = raw.partition("=")
        else:
            name, path = "default", raw
        if not name or not path:
            raise ValueError(f"Bad --adapter spec: {raw!r}")
        if name in out:
            raise ValueError(f"Duplicate adapter name: {name!r}")
        out[name] = path
    return out


class ModelService:
    """Holds the base model and any number of mounted PEFT adapters.

    Adapter switching uses ``PeftModel.set_adapter`` under a lock so two
    concurrent generations can't race on the active adapter.
    """

    def __init__(
        self,
        model_path: str,
        adapters: dict[str, str] | None = None,
        load_in_4bit: bool = False,
    ) -> None:
        self.model_path = model_path
        self.adapter_paths = adapters or {}
        self.load_in_4bit = load_in_4bit
        self.tokenizer: Any = None
        self.model: Any = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._adapter_lock = threading.Lock()
        self._active_adapter: str | None = None
        self.default_adapter: str | None = None

    def _build_kwargs(self) -> dict[str, Any]:
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if self.device == "cuda":
            kwargs["device_map"] = "auto"
        if self.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        return kwargs

    def load(self) -> None:
        logger.info(
            "Loading model",
            extra={
                "path": self.model_path,
                "device": self.device,
                "adapters": list(self.adapter_paths),
                "load_in_4bit": self.load_in_4bit,
            },
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **self._build_kwargs())

        if self.adapter_paths:
            from peft import PeftModel

            names = list(self.adapter_paths)
            first = names[0]
            self.model = PeftModel.from_pretrained(
                self.model, self.adapter_paths[first], adapter_name=first
            )
            for name in names[1:]:
                self.model.load_adapter(self.adapter_paths[name], adapter_name=name)
            self.default_adapter = first
            self._active_adapter = first

        self.model.eval()

    def list_adapters(self) -> list[AdapterInfo]:
        return [
            AdapterInfo(name=n, path=p, is_default=(n == self.default_adapter))
            for n, p in self.adapter_paths.items()
        ]

    @torch.no_grad()
    def generate(self, request: GenerateRequest) -> GenerateResponse:
        start = time.perf_counter()
        adapter_name = request.adapter or self.default_adapter

        if self.adapter_paths:
            if adapter_name not in self.adapter_paths:
                raise KeyError(f"Unknown adapter: {adapter_name!r}")
            with self._adapter_lock:
                if self._active_adapter != adapter_name:
                    self.model.set_adapter(adapter_name)
                    self._active_adapter = adapter_name

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
        label = adapter_name or "base"
        TOKENS.labels(label, "prompt").inc(prompt_tokens)
        TOKENS.labels(label, "completion").inc(completion_tokens)
        return GenerateResponse(
            text=text,
            finish_reason=finish,
            usage=GenerateUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            model=self.model_path,
            adapter=adapter_name,
            latency_ms=latency_ms,
        )


def create_app(
    model_path: str,
    adapter_path: str | None = None,
    adapters: list[str] | None = None,
    max_concurrency: int = 8,
    load_in_4bit: bool | None = None,
) -> FastAPI:
    """Create the FastAPI app.

    Compatibility:
      * ``adapter_path`` (singular) — single anonymous adapter.
      * ``adapters`` — list of ``name=path`` strings, takes precedence.
    """
    setup_logging()

    if adapters:
        adapter_map = _parse_adapters(adapters)
    elif adapter_path:
        adapter_map = {"default": adapter_path}
    else:
        adapter_map = {}

    if load_in_4bit is None:
        load_in_4bit = os.environ.get("LF_LOAD_IN_4BIT", "").lower() in {"1", "true", "yes"}

    service = ModelService(model_path, adapter_map, load_in_4bit=load_in_4bit)
    semaphore = asyncio.Semaphore(max_concurrency)
    auth_token = os.environ.get("API_AUTH_TOKEN")
    bearer = HTTPBearer(auto_error=False)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        service.load()
        yield

    app = FastAPI(title="lora-finetune inference", version="0.2.0", lifespan=lifespan)
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
            adapters=list(service.adapter_paths),
        )

    @app.get("/v1/adapters", response_model=AdaptersResponse, dependencies=[Depends(require_auth)])
    async def list_adapters() -> AdaptersResponse:
        return AdaptersResponse(adapters=service.list_adapters())

    @app.post("/v1/generate", response_model=GenerateResponse, dependencies=[Depends(require_auth)])
    async def generate(req: GenerateRequest, request: Request) -> GenerateResponse:
        adapter_label = req.adapter or service.default_adapter or "base"
        if semaphore.locked():
            REQUESTS.labels("generate", adapter_label, "throttled").inc()
        with LATENCY.labels("generate", adapter_label).time():
            async with semaphore:
                try:
                    resp = await asyncio.to_thread(service.generate, req)
                    REQUESTS.labels("generate", adapter_label, "ok").inc()
                    return resp
                except KeyError as e:
                    REQUESTS.labels("generate", adapter_label, "bad_request").inc()
                    raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e)) from e
                except Exception as e:
                    REQUESTS.labels("generate", adapter_label, "error").inc()
                    logger.exception("Generation failed")
                    raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(e)) from e

    return app
