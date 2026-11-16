#!/usr/bin/env bash
# End-to-end pipeline: data validation -> windowing -> inference ->
# root cause -> shadow decision -> evaluation -> figures.
#
# Modes:
#   bash scripts/run_pipeline.sh --mock [--inject-signal]   # no model needed
#   bash scripts/run_pipeline.sh                            # train + replay (full)
#   bash scripts/run_pipeline.sh --skip-train --checkpoint ckpt/best.ckpt
#   bash scripts/run_pipeline.sh --device mps
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"

MODE="full"
DEVICE="auto"
CKPT=""
MAXW="0"
INJECT=""
EPOCHS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mock) MODE="mock"; shift;;
    --inject-signal) INJECT="--inject-signal"; shift;;
    --skip-train) MODE="skip"; shift;;
    --checkpoint) CKPT="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --max-windows) MAXW="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

ARGS=( scripts/reproduce_all.py --device "$DEVICE" )
[[ "$MODE" == "mock" ]] && ARGS+=( --mock )
[[ -n "$INJECT" ]] && ARGS+=( "$INJECT" )
[[ "$MODE" == "skip" ]] && ARGS+=( --skip-train )
[[ -n "$CKPT" ]] && ARGS+=( --checkpoint "$CKPT" )
[[ "$MAXW" != "0" ]] && ARGS+=( --max-windows "$MAXW" )
[[ -n "$EPOCHS" ]] && ARGS+=( --epochs "$EPOCHS" )

echo "[run_pipeline] mode=$MODE device=$DEVICE"
exec "${PY}" "${ARGS[@]}"
