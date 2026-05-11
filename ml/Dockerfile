FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ml/requirements.txt /app/ml/requirements.txt
RUN pip install -r /app/ml/requirements.txt

COPY ml /app/ml

# Train once at build so the image is self-contained for benchmarks.
# (In prod the artifact is pulled from object storage at start-up.)
RUN python -m ml.training.train --rows 50000 --rounds 200 || true

ENV RANKER_MODEL_PATH=/app/ml/models/artifacts/ranker.txt
EXPOSE 8080

CMD ["uvicorn", "ml.serving.app:app", "--host", "0.0.0.0", "--port", "8080", \
     "--workers", "4", "--loop", "uvloop", "--http", "httptools"]
