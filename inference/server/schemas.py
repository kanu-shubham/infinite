from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class GenerateRequest(BaseModel):
    """Chat-style request. We mirror the subset of OpenAI's schema we need."""

    messages: list[Message] = Field(min_length=1)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stream: bool = True
    # Optional client-supplied id for log correlation; we generate one if absent.
    request_id: str | None = None
    # Stop sequences clamp runaway generations cheaply.
    stop: list[str] | None = None

    @field_validator("messages")
    @classmethod
    def _non_empty(cls, v: list[Message]) -> list[Message]:
        if not any(m.content.strip() for m in v):
            raise ValueError("messages must contain non-empty content")
        return v
