#!/usr/bin/env bash
# Single-node multi-GPU launcher using accelerate.
#
# Usage: scripts/train.sh configs/qlora_llama3_8b.yaml
set -euo pipefail

CONFIG=${1:-configs/qlora_llama3_8b.yaml}
NUM_GPUS=${NUM_GPUS:-$(python -c "import torch;print(torch.cuda.device_count())")}

exec accelerate launch \
    --num_processes "$NUM_GPUS" \
    --mixed_precision bf16 \
    -m lora_finetune.cli train --config "$CONFIG"
