#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: deploy.sh --model MODEL --video-dir DIR [options]

Required:
  --model MODEL          Hugging Face model ID or local model directory
  --video-dir DIR        Directory containing <question-id>.mp4 files

Options:
  --model-name NAME      Served model name (default: AdInsight-RL-Step300)
  --model-revision REV   Frozen Hugging Face model revision
  --log-dir DIR          Service log directory (default: <repo>/logs)
  --download-dir DIR     Hugging Face download/cache directory for vLLM
  --base-port PORT       First service port (default: 8240)
  --num-gpus N           Number of GPU services (default: all visible GPUs)
  --gpu-ids LIST         Comma-separated physical GPU IDs (default: CUDA_VISIBLE_DEVICES)
  --python PATH          Python executable (default: python)
  -h, --help             Show this help

The ADINSIGHT_MODEL, ADINSIGHT_VIDEO_DIR, ADINSIGHT_MODEL_NAME, BASE_PORT,
NUM_GPUS, and PYTHON_BIN environment variables remain supported as defaults.
EOF
}

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${ADINSIGHT_MODEL:-}"
VIDEO_DIR="${ADINSIGHT_VIDEO_DIR:-}"
MODEL_NAME="${ADINSIGHT_MODEL_NAME:-AdInsight-RL-Step300}"
MODEL_REVISION=""
LOG_DIR="$ROOT/logs"
DOWNLOAD_DIR=""
PYTHON="${PYTHON_BIN:-python}"
BASE_PORT="${BASE_PORT:-8240}"
NUM_GPUS="${NUM_GPUS:-}"
GPU_IDS_RAW="${CUDA_VISIBLE_DEVICES:-}"

while (( $# )); do
  case "$1" in
    --model) MODEL="${2:?Missing value for --model}"; shift 2 ;;
    --video-dir) VIDEO_DIR="${2:?Missing value for --video-dir}"; shift 2 ;;
    --model-name) MODEL_NAME="${2:?Missing value for --model-name}"; shift 2 ;;
    --model-revision) MODEL_REVISION="${2:?Missing value for --model-revision}"; shift 2 ;;
    --log-dir) LOG_DIR="${2:?Missing value for --log-dir}"; shift 2 ;;
    --download-dir) DOWNLOAD_DIR="${2:?Missing value for --download-dir}"; shift 2 ;;
    --base-port) BASE_PORT="${2:?Missing value for --base-port}"; shift 2 ;;
    --num-gpus) NUM_GPUS="${2:?Missing value for --num-gpus}"; shift 2 ;;
    --gpu-ids) GPU_IDS_RAW="${2:?Missing value for --gpu-ids}"; shift 2 ;;
    --python) PYTHON="${2:?Missing value for --python}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$MODEL" ]] || { echo "--model is required" >&2; exit 2; }
[[ -n "$VIDEO_DIR" ]] || { echo "--video-dir is required" >&2; exit 2; }
[[ -d "$VIDEO_DIR" ]] || { echo "Video directory not found: $VIDEO_DIR" >&2; exit 2; }
VIDEO_DIR="$(cd "$VIDEO_DIR" && pwd)"
if [[ -d "$MODEL" ]]; then MODEL="$(cd "$MODEL" && pwd)"; fi
if [[ -n "$DOWNLOAD_DIR" ]]; then mkdir -p "$DOWNLOAD_DIR"; DOWNLOAD_DIR="$(cd "$DOWNLOAD_DIR" && pwd)"; fi
mkdir -p "$LOG_DIR"
LOG_DIR="$(cd "$LOG_DIR" && pwd)"

GPU_LIST=()
if [[ -n "$GPU_IDS_RAW" ]]; then
  IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS_RAW"
else
  mapfile -t GPU_LINES < <(nvidia-smi -L)
  for ((i=0; i<${#GPU_LINES[@]}; i++)); do GPU_LIST+=("$i"); done
fi
if [[ -z "$NUM_GPUS" ]]; then NUM_GPUS="${#GPU_LIST[@]}"; fi
[[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]] || { echo "--num-gpus must be a positive integer" >&2; exit 2; }
(( NUM_GPUS <= ${#GPU_LIST[@]} )) || { echo "--num-gpus exceeds the available GPU ID list" >&2; exit 2; }
[[ "$BASE_PORT" =~ ^[0-9]+$ ]] || { echo "--base-port must be an integer" >&2; exit 2; }
command -v "$PYTHON" >/dev/null || { echo "Python executable not found: $PYTHON" >&2; exit 2; }

echo "[$(date -Is)] deploying $MODEL on $NUM_GPUS GPU(s)"

for ((i=0; i<NUM_GPUS; i++)); do
  PORT=$((BASE_PORT + i))
  if "$PYTHON" -c 'import socket, sys; s=socket.socket(); s.settimeout(1); raise SystemExit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)' "$PORT"; then
    echo "port $PORT is already in use; stop the existing service before deployment" >&2
    exit 1
  fi
done

PIDS=()
cleanup_failed_deployment() {
  if (( ${#PIDS[@]} )); then kill "${PIDS[@]}" 2>/dev/null || true; fi
}
trap cleanup_failed_deployment INT TERM
for ((i=0; i<NUM_GPUS; i++)); do
  PORT=$((BASE_PORT + i))
  LOG="$LOG_DIR/deploy_gpu${i}.log"
  EXTRA_ARGS=()
  if [[ -n "$DOWNLOAD_DIR" ]]; then EXTRA_ARGS+=(--download-dir "$DOWNLOAD_DIR"); fi
  if [[ -n "$MODEL_REVISION" ]]; then EXTRA_ARGS+=(--revision "$MODEL_REVISION"); fi

  CUDA_VISIBLE_DEVICES="${GPU_LIST[$i]}" nohup "$PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name "$MODEL_NAME" \
    --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size 1 --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    --limit-mm-per-prompt '{"video":1,"image":0}' \
    --allowed-local-media-path "$VIDEO_DIR" \
    --mm-processor-cache-gb 0 --no-enable-prefix-caching \
    --max-num-seqs 4 --enable-chunked-prefill \
    --reasoning-parser qwen3 --trust-remote-code \
    "${EXTRA_ARGS[@]}" >"$LOG" 2>&1 &
  PIDS+=("$!")
  echo "  GPU $i -> port $PORT (PID ${PIDS[-1]})"
done

echo "[$(date -Is)] waiting for services to be ready..."
READY=0
for ((i=0; i<NUM_GPUS; i++)); do
  PORT=$((BASE_PORT + i))
  for _ in $(seq 1 120); do
    if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
      echo "  GPU $i service exited before becoming ready; check $LOG_DIR/deploy_gpu${i}.log" >&2
      break
    fi
    if curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q "$MODEL_NAME"; then
      echo "  GPU $i ready on port $PORT"
      READY=$((READY + 1))
      break
    fi
    sleep 5
  done
done

if (( READY != NUM_GPUS )); then
  echo "[$(date -Is)] only $READY/$NUM_GPUS ready; check $LOG_DIR" >&2
  cleanup_failed_deployment
  exit 1
fi
for pid in "${PIDS[@]}"; do
  kill -0 "$pid" 2>/dev/null || { echo "a deployed service exited unexpectedly" >&2; cleanup_failed_deployment; exit 1; }
done
trap - INT TERM

echo "[$(date -Is)] all $NUM_GPUS service(s) ready"
for ((i=0; i<NUM_GPUS; i++)); do
  echo "  http://127.0.0.1:$((BASE_PORT+i))/v1"
done
