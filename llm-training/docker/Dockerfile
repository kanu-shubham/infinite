# syntax=docker/dockerfile:1.7
# CUDA 12.1 + PyTorch 2.3 base with Ubuntu 22.04.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-venv \
        git git-lfs curl ca-certificates build-essential \
        libopenmpi-dev openmpi-bin ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /workspace

# Install deps first for caching. Copy only the manifest.
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel \
    && pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121 \
    && pip install -e ".[dev]" || pip install -e "."

# Now copy source.
COPY . .
RUN pip install -e .

# Non-root user for safer runs (override with --user 0 if needed).
RUN useradd --create-home --shell /bin/bash trainer \
    && chown -R trainer:trainer /workspace
USER trainer

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import llm_train; print('ok')" || exit 1

CMD ["bash"]
