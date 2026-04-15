# ── Stage 1: install dependencies ────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# PyTorch CPU-only wheel is much smaller than the default (no CUDA drivers needed)
RUN pip install --no-cache-dir --user torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# ── Stage 2: lean production image ────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="mlops-demo"
LABEL description="Hotel Recommendation API (Two-Tower + XGBoost)"

WORKDIR /app

COPY --from=builder /root/.local /root/.local

COPY config.yaml .
COPY src/ src/
COPY models/ models/

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8001

RUN adduser --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]
