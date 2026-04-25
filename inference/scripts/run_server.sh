#!/usr/bin/env bash
# Usage: scripts/run_server.sh configs/baseline.env
set -euo pipefail
cfg="${1:-configs/baseline.env}"
set -a
# shellcheck disable=SC1090
. "$cfg"
set +a
exec uvicorn server.main:app \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8000}" \
  --log-level "${LOG_LEVEL:-info}" \
  --workers 1
