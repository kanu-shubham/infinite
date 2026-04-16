# Dockerfile for the Hotel Price Prediction API
# ----------------------------------------------
# Multi-stage build:
#   stage 1 (builder) – install Python deps into a venv
#   stage 2 (runtime) – copy only the venv + source, no build tools
# This keeps the final image small (~300 MB vs ~1 GB single-stage).

# ── Stage 1: build ────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching.
# If requirements.txt doesn't change, this layer is reused.
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY . .

# Port the FastAPI app listens on
EXPOSE 8000

# Health check: Kubernetes calls GET /health every 30s.
# If it returns non-200 three times in a row, the pod is restarted.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Non-root user for security (never run containers as root in prod)
RUN useradd -m appuser
USER appuser

CMD ["uvicorn", "serving.predict:app", "--host", "0.0.0.0", "--port", "8000"]
