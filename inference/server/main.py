"""FastAPI app exposing /v1/chat (SSE), /healthz, /readyz, /metrics.

The hot path is intentionally simple:

    request -> validate -> admission slot -> engine.stream -> SSE

Everything else (timeouts, cancellation, metrics, structured logs) is
layered as middleware/wrappers so the streaming code reads top-to-bottom.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager

import orjson
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from . import logging_setup, metrics
from .admission import Admission, Overloaded
from .config import settings
from .engine import Engine
from .schemas import GenerateRequest

log = logging.getLogger("inference")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging_setup.configure(settings.log_level)
    app.state.engine = Engine(settings)
    app.state.admission = Admission(
        max_inflight=settings.max_inflight,
        max_queue=settings.max_queue,
    )
    await app.state.engine.start()
    log.info("server ready", extra={"port": settings.port})
    try:
        yield
    finally:
        log.info("server draining")
        await app.state.engine.stop()


app = FastAPI(lifespan=lifespan, default_response_class=JSONResponse)


@app.get("/healthz")
async def healthz() -> dict:
    """Liveness: process is up. Cheap; no engine touch."""
    return {"status": "ok"}


@app.get("/readyz")
async def readyz() -> Response:
    """Readiness: engine has loaded weights and can accept requests.

    Kept separate from liveness so a slow model load doesn't get the
    pod killed by the kubelet during startup.
    """
    if not app.state.engine.ready:
        return JSONResponse({"status": "not_ready"}, status_code=503)
    return JSONResponse({"status": "ready"})


@app.get("/metrics")
async def metrics_endpoint() -> Response:
    body, content_type = metrics.render()
    return Response(content=body, media_type=content_type)


def _sse(event: dict) -> bytes:
    return b"data: " + orjson.dumps(event) + b"\n\n"


@app.post("/v1/chat")
async def chat(req: GenerateRequest, http_req: Request) -> Response:
    if req.max_tokens > settings.max_output_tokens:
        raise HTTPException(
            status_code=400,
            detail=f"max_tokens exceeds server cap ({settings.max_output_tokens})",
        )

    rid = req.request_id or f"req-{uuid.uuid4().hex[:12]}"
    admission: Admission = app.state.admission
    engine: Engine = app.state.engine

    try:
        admission_cm = admission.slot()
        await admission_cm.__aenter__()
    except Overloaded as ov:
        metrics.requests_total.labels(outcome="rejected").inc()
        return JSONResponse(
            {"error": "overloaded", "request_id": rid},
            status_code=429,
            headers={"Retry-After": f"{ov.retry_after_s:.1f}"},
        )

    started = time.monotonic()

    async def gen():
        first_token_at: float | None = None
        last_token_at: float | None = None
        token_count = 0
        outcome = "ok"
        try:
            stream = engine.stream(
                [m.model_dump() for m in req.messages],
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                stop=req.stop,
                request_id=rid,
            )
            # Wrap the stream in a per-request timeout so a stuck request
            # cannot consume an admission slot indefinitely.
            async def with_timeout():
                async for delta in stream:
                    yield delta

            async for delta in _timed(with_timeout(), settings.request_timeout_s):
                if await http_req.is_disconnected():
                    # Client gave up; cancelling the iterator triggers
                    # engine.abort() in Engine.stream().
                    raise asyncio.CancelledError()

                now = delta.monotonic_s
                if first_token_at is None:
                    first_token_at = now
                    metrics.ttft_seconds.observe(now - started)
                else:
                    assert last_token_at is not None
                    metrics.itl_seconds.observe(now - last_token_at)
                last_token_at = now
                token_count += 1
                yield _sse({"type": "delta", "text": delta.text, "index": delta.index})

            yield _sse({"type": "done", "tokens": token_count})
        except asyncio.TimeoutError:
            outcome = "timeout"
            yield _sse({"type": "error", "error": "timeout"})
        except asyncio.CancelledError:
            outcome = "client_cancelled"
            # Don't try to send more bytes; client is gone.
            raise
        except Exception as e:  # noqa: BLE001
            outcome = "error"
            log.exception("generation failed", extra={"request_id": rid})
            yield _sse({"type": "error", "error": type(e).__name__})
        finally:
            metrics.requests_total.labels(outcome=outcome).inc()
            metrics.output_tokens.observe(token_count)
            metrics.e2e_seconds.observe(time.monotonic() - started)
            await admission_cm.__aexit__(None, None, None)
            log.info(
                "request complete",
                extra={
                    "request_id": rid,
                    "outcome": outcome,
                    "tokens": token_count,
                    "ttft_ms": (
                        round((first_token_at - started) * 1000, 2)
                        if first_token_at else None
                    ),
                    "e2e_ms": round((time.monotonic() - started) * 1000, 2),
                },
            )

    return StreamingResponse(gen(), media_type="text/event-stream")


async def _timed(aiter, timeout_s: float):
    """Yield from `aiter` but raise TimeoutError if any single step stalls.

    We bound *per-step* rather than total elapsed: a long but healthy
    generation should not be killed, but a hung step should.
    """
    while True:
        try:
            item = await asyncio.wait_for(aiter.__anext__(), timeout=timeout_s)
        except StopAsyncIteration:
            return
        yield item
