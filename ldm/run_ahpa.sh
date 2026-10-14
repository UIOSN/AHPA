#!/usr/bin/env bash

set -euo pipefail

MODE="${MODE:-ahpa}"
NGPU="${NGPU:-8}"
MODEL="${MODEL:-SiT-XL/2}"
DATA_DIR="${DATA_DIR:-../data}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-400000}"
EXP_NAME="${EXP_NAME:-${MODE}-sit-xl}"

case "$MODE" in
  ahpa|sra2) ;;
  *) echo "MODE must be 'ahpa' or 'sra2'" >&2; exit 2 ;;
esac

cd "$(dirname "$0")"

accelerate launch --num_processes "$NGPU" --main_process_port 29501 train.py \
  --config "configs/${MODE}.yaml" \
  --model "$MODEL" \
  --encoder-depth 7 \
  --data-dir "$DATA_DIR" \
  --exp-name "$EXP_NAME" \
  --max-train-steps "$MAX_TRAIN_STEPS" \
  --learning-rate 2e-4 \
  --max-grad-norm 2.0 \
  --batch-size 256
