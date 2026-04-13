# ── Stage 1: build dependencies ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install only production deps into a user-local directory
# so we can copy them cleanly into the final image
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# ── Stage 2: lean production image ────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="mlops-demo"
LABEL description="Hotel Price Prediction API"

WORKDIR /app

# Copy installed packages from builder (avoids pip in prod image)
COPY --from=builder /root/.local /root/.local

# Copy only what the API needs at runtime
COPY config.yaml .
COPY src/       src/
# Model artifact is mounted via docker-compose volume (or baked in for prod)
COPY models/    models/

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Use a non-root user in production
RUN adduser --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
