#!/usr/bin/env bash
set -euo pipefail

MODEL_TYPE="${1:?usage: train_8B.sh <llama|qwen> <initialized-model> <alpha> [output-dir]}"
MODEL_PATH="${2:?missing initialized model path}"
ALPHA="${3:?missing alpha value}"
OUTPUT_DIR="${4:-outputs/${MODEL_TYPE}/a${ALPHA}}"

case "$MODEL_TYPE" in
  llama)
    DEEPSPEED_CONFIG="configs/deepspeed_llama.json"
    LEARNING_RATE="1e-5"
    SEQUENCE_LENGTH="8192"
    MAX_STEPS="500"
    MICRO_BATCH_SIZE="4"
    GRADIENT_ACCUMULATION="8"
    ;;
  qwen)
    DEEPSPEED_CONFIG="configs/deepspeed_qwen.json"
    LEARNING_RATE="5e-5"
    SEQUENCE_LENGTH="32768"
    MAX_STEPS="250"
    MICRO_BATCH_SIZE="1"
    GRADIENT_ACCUMULATION="16"
    ;;
  *)
    echo "MODEL_TYPE must be llama or qwen" >&2
    exit 2
    ;;
esac

export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export WANDB_MODE="${WANDB_MODE:-offline}"
NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29592}"

# Both reference runs process 131,072,000 tokens on one H200 GPU:
# Llama: 500 * 1 GPU * micro-batch 4 * accumulation 8 * 8192 tokens.
# Qwen:  250 * 1 GPU * micro-batch 1 * accumulation 16 * 32768 tokens.
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" scripts/train.py \
  --model "$MODEL_PATH" \
  --model-type "$MODEL_TYPE" \
  --alpha "$ALPHA" \
  --temperature 2.0 \
  --dataset-path data/redpajama \
  --sequence-length "$SEQUENCE_LENGTH" \
  --max-steps "$MAX_STEPS" \
  --per-device-train-batch-size "$MICRO_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION" \
  --learning-rate "$LEARNING_RATE" \
  --deepspeed "$DEEPSPEED_CONFIG" \
  --output-dir "$OUTPUT_DIR"
