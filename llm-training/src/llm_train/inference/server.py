"""Minimal OpenAI-compatible inference server (FastAPI).

Not meant to replace vLLM / TGI in high-throughput production — it's a
thin wrapper for local smoke tests and integration into demos.
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from llm_train.evaluation.generate import (
    GenerationParams,
    batch_generate,
    load_for_inference,
)
from llm_train.utils.logging import get_logger, setup_logging

log = get_logger(__name__)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    do_sample: bool = True


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    model: str
    choices: list[ChatChoice]


_state: dict[str, Any] = {"model": None, "tokenizer": None, "model_name": ""}


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    model_path = os.environ["LLM_TRAIN_MODEL_PATH"]
    log.info("Loading model %s", model_path)
    model, tokenizer = load_for_inference(model_path)
    _state["model"] = model
    _state["tokenizer"] = tokenizer
    _state["model_name"] = model_path
    yield
    _state.clear()


app = FastAPI(title="llm-train inference", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    ok = _state.get("model") is not None
    return {"status": "ok" if ok else "loading", "model": _state.get("model_name")}


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    model = _state.get("model")
    tokenizer = _state.get("tokenizer")
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    messages = [m.model_dump() for m in req.messages]
    completions = batch_generate(
        model,
        tokenizer,
        [messages],
        params=GenerationParams(
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            do_sample=req.do_sample,
        ),
    )

    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        model=_state["model_name"],
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content=completions[0].strip()),
                finish_reason="stop",
            )
        ],
    )
