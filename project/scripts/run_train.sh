#!/usr/bin/env bash
# Dedicated training entry point for Linux CUDA / macOS MPS / CPU.
#
# Examples:
#   bash scripts/run_train.sh                                  # defaults from config
#   bash scripts/run_train.sh --device mps --epochs 10
#   bash scripts/run_train.sh --device cuda --epochs 20 --batch-size 256
#   bash scripts/run_train.sh --model TranAD --tag base        # baseline run
#   bash scripts/run_train.sh --max-train-windows 500 --epochs 1   # smoke test
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"

MODEL=""
DEVICE="auto"
EPOCHS=""
BATCH=""
LR=""
SEED=""
TAG=""
MAXW="0"
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --batch-size) BATCH="$2"; shift 2;;
    --lr|--learning-rate) LR="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    --max-train-windows) MAXW="$2"; shift 2;;
    *) EXTRA+=("$1"); shift;;
  esac
done

ARGS=( -m src.models.train --device "$DEVICE" )
[[ -n "$MODEL" ]] && ARGS+=( --model "$MODEL" )
[[ -n "$EPOCHS" ]] && ARGS+=( --epochs "$EPOCHS" )
[[ -n "$BATCH" ]] && ARGS+=( --batch-size "$BATCH" )
[[ -n "$LR" ]] && ARGS+=( --lr "$LR" )
[[ -n "$SEED" ]] && ARGS+=( --seed "$SEED" )
[[ "$MAXW" != "0" ]] && ARGS+=( --max-train-windows "$MAXW" )
[[ ${#EXTRA[@]} -gt 0 ]] && ARGS+=( "${EXTRA[@]}" )

LOG_DIR="artifacts/train_logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/train_${TAG:-${MODEL:-cfg}}_${STAMP}.log"

echo "[run_train] python=${PY} args: ${ARGS[*]}"
echo "[run_train] log -> ${LOG_FILE}"
"${PY}" "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"
