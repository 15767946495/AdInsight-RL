#!/usr/bin/env bash
set -euo pipefail

# Deploy the AdInsight-RL model with vLLM.
# Auto-detects available GPUs. One vLLM instance per GPU, TP=1.

MODEL="${ADINSIGHT_MODEL:?Set ADINSIGHT_MODEL to HF model ID or local path}"
VIDEO_DIR="${ADINSIGHT_VIDEO_DIR:?Set ADINSIGHT_VIDEO_DIR}"
MODEL_NAME="${ADINSIGHT_MODEL_NAME:-AdInsight-RL-Step300}"
PYTHON="${PYTHON_BIN:-python}"
BASE_PORT="${BASE_PORT:-8240}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)}"

ROOT="${ADINSIGHT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

echo "[$(date -Is)] deploying $MODEL on $NUM_GPUS GPU(s)"

PIDS=()
for i in $(seq 0 $((NUM_GPUS-1))); do
  PORT=$((BASE_PORT + i))
  LOG="$LOG_DIR/deploy_gpu${i}.log"

  CUDA_VISIBLE_DEVICES="$i" nohup "$PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name "$MODEL_NAME" \
    --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size 1 --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    --limit-mm-per-prompt '{"video":1,"image":0}' \
    --allowed-local-media-path "$VIDEO_DIR" \
    --mm-processor-cache-gb 0 --no-enable-prefix-caching \
    --max-num-seqs 4 --enable-chunked-prefill \
    --reasoning-parser qwen3 --trust-remote-code \
    >"$LOG" 2>&1 &
  PID=$!
  PIDS+=("$PID")
  echo "  GPU $i -> port $PORT (PID $PID)"
done

echo "[$(date -Is)] waiting for services to be ready..."
READY=0
for i in $(seq 0 $((NUM_GPUS-1))); do
  PORT=$((BASE_PORT + i))
  for _ in $(seq 1 120); do
    if curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q "$MODEL_NAME"; then
      echo "  GPU $i ready on port $PORT"
      READY=$((READY + 1))
      break
    fi
    sleep 5
  done
done

if (( READY == NUM_GPUS )); then
  echo "[$(date -Is)] all $NUM_GPUS service(s) ready"
  echo "endpoints:"
  for i in $(seq 0 $((NUM_GPUS-1))); do
    echo "  http://127.0.0.1:$((BASE_PORT+i))/v1"
  done
else
  echo "[$(date -Is)] only $READY/$NUM_GPUS ready; check $LOG_DIR/" >&2
  exit 1
fi
