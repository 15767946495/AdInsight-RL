#!/usr/bin/env bash
set -euo pipefail

# Generate ASR transcripts using faster-whisper.
# Auto-detects available GPUs and runs one shard per GPU.

ROOT="${ADINSIGHT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON="${PYTHON_BIN:-python}"
VIDEOS="${ADINSIGHT_VIDEO_DIR:?Set ADINSIGHT_VIDEO_DIR}"
WHISPER_MODEL="${WHISPER_MODEL:-large-v3-turbo}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)}"

OUT_DIR="$ROOT/data/asr_raw"
FINAL="$ROOT/data/asr_final.jsonl"
mkdir -p "$OUT_DIR"

echo "[$(date -Is)] transcribing on $NUM_GPUS GPU(s) with faster-whisper"
PIDS=()
for i in $(seq 0 $((NUM_GPUS-1))); do
  CUDA_VISIBLE_DEVICES="$i" "$PYTHON" "$ROOT/scripts/generate_asr.py" \
    --videos "$VIDEOS" --output "$OUT_DIR/shard_$i.jsonl" \
    --model "$WHISPER_MODEL" --shard "$i" --num-shards "$NUM_GPUS" \
    --resume \
    >"$OUT_DIR/shard_$i.log" 2>&1 &
  PIDS+=("$!")
done

FAILED=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=$((FAILED+1)); done
if (( FAILED > 0 )); then
  echo "[$(date -Is)] $FAILED ASR shards failed" >&2
  exit 1
fi

echo "[$(date -Is)] merging + post-processing"
MERGED="$OUT_DIR/merged.jsonl"
cat "$OUT_DIR"/shard_*.jsonl > "$MERGED"
"$PYTHON" "$ROOT/scripts/postprocess_asr.py" \
  --input "$MERGED" --videos "$VIDEOS" --output "$FINAL"

echo "[$(date -Is)] done: $FINAL ($(wc -l < "$FINAL") rows)"
