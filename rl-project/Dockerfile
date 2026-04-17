# Dockerfile — serve the trained PPO policy as a REST API.
#
# Build:
#   docker build -t rl-inventory .
# Run:
#   docker run --rm -p 8000:8000 -e RL_CHECKPOINT=/app/runs/ppo_inventory/best.pt rl-inventory

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt setup.py ./
COPY src ./src

RUN pip install --upgrade pip \
 && pip install -r requirements.txt \
 && pip install -e .

COPY scripts ./scripts

EXPOSE 8000
CMD ["uvicorn", "scripts.serve:app", "--host", "0.0.0.0", "--port", "8000"]
