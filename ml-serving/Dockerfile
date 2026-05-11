# syntax=docker/dockerfile:1
#
# Two-stage build:
#   1. builder  — install deps + train the model (heavy, cached by Docker)
#   2. runtime  — lean image with only what's needed to serve
#
# Training at build time means the image is fully self-contained.
# In production you'd pull model artefacts from a model registry (MLflow,
# S3, Vertex AI Model Registry) at startup instead.

# ── Stage 1: train ───────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .
# Train writes model.pkl to the working directory
RUN python train.py


# ── Stage 2: serve ───────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the trained model from the builder stage
COPY --from=builder /build/model.pkl ./model.pkl

# Copy the application
COPY app/ ./app/

EXPOSE 8000

# uvicorn with 4 workers: each is a separate process with its own GIL.
# This gives true CPU parallelism for inference-heavy workloads without
# needing to worry about thread safety of the model itself.
#
# For I/O-bound services (model behind a gRPC call), 1 worker + many
# async coroutines is better; multiple workers waste memory.
#
# --timeout-keep-alive: keep HTTP/1.1 connections open longer under load
# to avoid TCP handshake overhead at high RPS.
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--timeout-keep-alive", "30"]
