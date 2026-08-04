#!/usr/bin/env bash
set -euo pipefail

# Run inference across all deployed services.
# Auto-detects GPU count to match the deployment.

ROOT="${ADINSIGHT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON="${PYTHON_BIN:-python}"
MODEL_NAME="${ADINSIGHT_MODEL_NAME:-AdInsight-RL-Step300}"
BASE_PORT="${BASE_PORT:-8240}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)}"

QUESTIONS="$ROOT/data/MAC_QA.jsonl"
ASR="${ADINSIGHT_ASR:-$ROOT/data/asr_final.jsonl}"
VIDEOS="${ADINSIGHT_VIDEO_DIR:?Set ADINSIGHT_VIDEO_DIR}"
OUT_DIR="$ROOT/outputs"
FINAL="$OUT_DIR/submission.jsonl"
mkdir -p "$OUT_DIR"

echo "[$(date -Is)] starting $NUM_GPUS inference shards"
PIDS=()
for i in $(seq 0 $((NUM_GPUS-1))); do
  "$PYTHON" "$ROOT/scripts/inference.py" \
    --base-url "http://127.0.0.1:$((BASE_PORT+i))/v1" \
    --model "$MODEL_NAME" --videos "$VIDEOS" \
    --questions "$QUESTIONS" --asr "$ASR" \
    --output "$OUT_DIR/shard_$i.jsonl" \
    --shard "$i" --num-shards "$NUM_GPUS" \
    --workers 4 --resume \
    >"$OUT_DIR/shard_$i.log" 2>&1 &
  PIDS+=("$!")
done

FAILED=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=$((FAILED+1)); done
if (( FAILED > 0 )); then
  echo "[$(date -Is)] $FAILED shards failed" >&2
  exit 1
fi

echo "[$(date -Is)] merging $NUM_GPUS shards"
ARGS=(--questions "$QUESTIONS" --output "$FINAL")
for i in $(seq 0 $((NUM_GPUS-1))); do
  ARGS+=(--shard "$OUT_DIR/shard_$i.jsonl")
done
"$PYTHON" "$ROOT/scripts/merge_shards.py" "${ARGS[@]}"

echo "[$(date -Is)] done: $FINAL ($(wc -l < "$FINAL") rows)"
