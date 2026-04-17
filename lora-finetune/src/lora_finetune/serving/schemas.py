"""Request/response schemas for the inference API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class GenerateRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)
    max_new_tokens: int = Field(256, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=0, le=500)
    repetition_penalty: float = Field(1.0, ge=0.5, le=2.0)
    stop: list[str] | None = None
    stream: bool = False


class GenerateUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class GenerateResponse(BaseModel):
    text: str
    finish_reason: Literal["stop", "length"]
    usage: GenerateUsage
    model: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model: str
    device: str
