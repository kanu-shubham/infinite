#!/usr/bin/env bash
# Export a (merged) HuggingFace model to GGUF for llama.cpp / Ollama / LM Studio.
#
# GGUF is the file format used by llama.cpp's CPU/Metal inference. The
# conversion ships with llama.cpp itself; this script just orchestrates it.
#
# Usage:
#   scripts/export_gguf.sh <hf_model_dir> <out_dir> [quant]
#
#   quant: f16 (default) | q8_0 | q5_k_m | q4_k_m | q4_0 | q3_k_m | q2_k
#
# Examples:
#   scripts/export_gguf.sh outputs/merged exports/llama3.gguf q4_k_m
#
# Requires llama.cpp checked out at $LLAMA_CPP (default: ../llama.cpp).
set -euo pipefail

MODEL_DIR=${1:?"hf model directory required"}
OUT_DIR=${2:?"output directory required"}
QUANT=${3:-f16}
LLAMA_CPP=${LLAMA_CPP:-../llama.cpp}

if [[ ! -d "$LLAMA_CPP" ]]; then
    echo "llama.cpp not found at $LLAMA_CPP. Set LLAMA_CPP or clone it:"
    echo "  git clone https://github.com/ggerganov/llama.cpp $LLAMA_CPP"
    echo "  (cd $LLAMA_CPP && make)"
    exit 1
fi

mkdir -p "$OUT_DIR"
F16_FILE="$OUT_DIR/model-f16.gguf"

echo "[gguf] Converting $MODEL_DIR -> $F16_FILE"
python "$LLAMA_CPP/convert_hf_to_gguf.py" "$MODEL_DIR" --outfile "$F16_FILE" --outtype f16

if [[ "$QUANT" == "f16" ]]; then
    echo "[gguf] Done: $F16_FILE"
    exit 0
fi

QUANT_FILE="$OUT_DIR/model-${QUANT}.gguf"
echo "[gguf] Quantizing -> $QUANT_FILE ($QUANT)"
"$LLAMA_CPP/build/bin/llama-quantize" "$F16_FILE" "$QUANT_FILE" "$QUANT"

echo "[gguf] Done: $QUANT_FILE"
