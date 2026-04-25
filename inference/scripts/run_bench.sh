#!/usr/bin/env bash
# Run the three configurations end-to-end. Assumes the server is restarted
# externally between configs (it must be -- engine flags are immutable
# after boot). Pass the label as $1; it gates which workload/output the
# benchmark uses. Example:
#   scripts/run_bench.sh baseline
#   scripts/run_bench.sh apc
#   scripts/run_bench.sh apc_chunked
set -euo pipefail
label="${1:?usage: run_bench.sh <baseline|apc|apc_chunked>}"
url="${URL:-http://localhost:8000/v1/chat}"
out_dir="${OUT_DIR:-bench/results}"

mkdir -p "$out_dir"

# Run shared_system to expose APC; mixed to expose chunked-prefill.
python -m bench.benchmark --url "$url" --workload shared_system \
    --n 200 --concurrency 16 --warmup 8 \
    --label "$label-shared" --out "$out_dir/$label-shared.json"

python -m bench.benchmark --url "$url" --workload mixed \
    --n 200 --concurrency 24 --warmup 8 \
    --label "$label-mixed" --out "$out_dir/$label-mixed.json"
