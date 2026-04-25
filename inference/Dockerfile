# vLLM ships a CUDA-ready base image; reuse it instead of pinning CUDA
# ourselves. The python deps in requirements.txt are kept thin so layer
# rebuilds stay fast.
FROM vllm/vllm-openai:v0.6.3.post1

WORKDIR /app
COPY requirements.txt /app/requirements.txt
# vLLM is already installed in the base image; install the rest.
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY server /app/server
COPY bench  /app/bench
COPY configs /app/configs
COPY scripts /app/scripts
RUN chmod +x /app/scripts/*.sh

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_DISABLE_TELEMETRY=1

EXPOSE 8000
# Default to baseline; override with `--env-file configs/apc.env`.
CMD ["bash", "scripts/run_server.sh", "configs/baseline.env"]
