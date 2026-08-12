#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_inference.sh --videos DIR --asr FILE [options]

Options:
  --questions FILE       Question JSONL (default: data/MAC_QA.jsonl)
  --asr FILE             ASR JSONL (required unless ADINSIGHT_ASR is set)
  --videos DIR           Directory containing <question-id>.mp4 files
  --output-dir DIR       Shard output directory (default: outputs)
  --output FILE          Final submission JSONL (default: <output-dir>/submission.jsonl)
  --manifest FILE        Reproduction manifest (default: <output>.manifest.json)
  --system-prompt FILE   System prompt (default: prompts/final_answer.txt)
  --model-name NAME      Served model name (default: AdInsight-RL-Step300)
  --model-source MODEL   HF model ID or local model directory used by deployment
  --model-revision REV   Frozen model revision recorded in the manifest
  --base-port PORT       First vLLM port (default: 8240)
  --num-gpus N           Number of inference shards (default: all visible GPUs)
  --workers N            Concurrent requests per service (default: 4)
  --max-tokens N         Maximum completion tokens (default: 384)
  --temperature VALUE    Sampling temperature (default: 0.2)
  --top-p VALUE          Top-p sampling value (default: 0.9)
  --seed N               Random seed (default: 42)
  --resume               Resume matching shard outputs
  --no-resume            Start fresh (default)
  --python PATH          Python executable (default: python)
  -h, --help             Show this help
EOF
}

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON_BIN:-python}"
MODEL_NAME="${ADINSIGHT_MODEL_NAME:-AdInsight-RL-Step300}"
MODEL_SOURCE="${ADINSIGHT_MODEL:-}"
MODEL_REVISION=""
BASE_PORT="${BASE_PORT:-8240}"
NUM_GPUS="${NUM_GPUS:-}"
QUESTIONS="$ROOT/data/MAC_QA.jsonl"
ASR="${ADINSIGHT_ASR:-}"
VIDEOS="${ADINSIGHT_VIDEO_DIR:-}"
OUT_DIR="$ROOT/outputs"
FINAL=""
MANIFEST=""
SYSTEM_PROMPT="$ROOT/prompts/final_answer.txt"
WORKERS=4
MAX_TOKENS=384
TEMPERATURE=0.2
TOP_P=0.9
SEED=42
RESUME=0

while (( $# )); do
  case "$1" in
    --questions) QUESTIONS="${2:?Missing value for --questions}"; shift 2 ;;
    --asr) ASR="${2:?Missing value for --asr}"; shift 2 ;;
    --videos) VIDEOS="${2:?Missing value for --videos}"; shift 2 ;;
    --output-dir) OUT_DIR="${2:?Missing value for --output-dir}"; shift 2 ;;
    --output) FINAL="${2:?Missing value for --output}"; shift 2 ;;
    --manifest) MANIFEST="${2:?Missing value for --manifest}"; shift 2 ;;
    --system-prompt) SYSTEM_PROMPT="${2:?Missing value for --system-prompt}"; shift 2 ;;
    --model-name) MODEL_NAME="${2:?Missing value for --model-name}"; shift 2 ;;
    --model-source) MODEL_SOURCE="${2:?Missing value for --model-source}"; shift 2 ;;
    --model-revision) MODEL_REVISION="${2:?Missing value for --model-revision}"; shift 2 ;;
    --base-port) BASE_PORT="${2:?Missing value for --base-port}"; shift 2 ;;
    --num-gpus) NUM_GPUS="${2:?Missing value for --num-gpus}"; shift 2 ;;
    --workers) WORKERS="${2:?Missing value for --workers}"; shift 2 ;;
    --max-tokens) MAX_TOKENS="${2:?Missing value for --max-tokens}"; shift 2 ;;
    --temperature) TEMPERATURE="${2:?Missing value for --temperature}"; shift 2 ;;
    --top-p) TOP_P="${2:?Missing value for --top-p}"; shift 2 ;;
    --seed) SEED="${2:?Missing value for --seed}"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --no-resume) RESUME=0; shift ;;
    --python) PYTHON="${2:?Missing value for --python}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$VIDEOS" ]] || { echo "--videos is required" >&2; exit 2; }
[[ -n "$ASR" ]] || { echo "--asr is required" >&2; exit 2; }
[[ -n "$MODEL_SOURCE" ]] || { echo "--model-source is required" >&2; exit 2; }
[[ -d "$VIDEOS" ]] || { echo "Video directory not found: $VIDEOS" >&2; exit 2; }
for file in "$QUESTIONS" "$ASR" "$SYSTEM_PROMPT"; do [[ -f "$file" ]] || { echo "File not found: $file" >&2; exit 2; }; done
VIDEOS="$(cd "$VIDEOS" && pwd)"
QUESTIONS="$(cd "$(dirname "$QUESTIONS")" && pwd)/$(basename "$QUESTIONS")"
ASR="$(cd "$(dirname "$ASR")" && pwd)/$(basename "$ASR")"
SYSTEM_PROMPT="$(cd "$(dirname "$SYSTEM_PROMPT")" && pwd)/$(basename "$SYSTEM_PROMPT")"
if [[ -z "$NUM_GPUS" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
    NUM_GPUS="${#GPU_LIST[@]}"
  else
    mapfile -t GPU_LINES < <(nvidia-smi -L)
    NUM_GPUS="${#GPU_LINES[@]}"
  fi
fi
[[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]] || { echo "--num-gpus must be a positive integer" >&2; exit 2; }
[[ "$BASE_PORT" =~ ^[0-9]+$ ]] || { echo "--base-port must be an integer" >&2; exit 2; }
command -v "$PYTHON" >/dev/null || { echo "Python executable not found: $PYTHON" >&2; exit 2; }
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
FINAL="${FINAL:-$OUT_DIR/submission.jsonl}"
mkdir -p "$(dirname "$FINAL")"
FINAL="$(cd "$(dirname "$FINAL")" && pwd)/$(basename "$FINAL")"
MANIFEST="${MANIFEST:-${FINAL%.jsonl}.manifest.json}"
mkdir -p "$(dirname "$MANIFEST")"
MANIFEST="$(cd "$(dirname "$MANIFEST")" && pwd)/$(basename "$MANIFEST")"
for input in "$QUESTIONS" "$ASR" "$SYSTEM_PROMPT"; do
  [[ "$FINAL" != "$input" ]] || { echo "--output must not overwrite an input file" >&2; exit 2; }
  [[ "$MANIFEST" != "$input" ]] || { echo "--manifest must not overwrite an input file" >&2; exit 2; }
done
[[ "$FINAL" != "$MANIFEST" ]] || { echo "--output and --manifest must be different" >&2; exit 2; }

"$PYTHON" "$ROOT/scripts/preflight.py" inference --questions "$QUESTIONS" --videos "$VIDEOS" --asr "$ASR"

RUN_METADATA="$OUT_DIR/run_metadata.json"
METADATA_ARGS=(
  --kind inference --questions "$QUESTIONS" --videos "$VIDEOS"
  --model "$MODEL_SOURCE" --model-revision "$MODEL_REVISION" --output "$RUN_METADATA"
  --file "asr=$ASR" --file "system_prompt=$SYSTEM_PROMPT"
  --setting "served_model_name=$MODEL_NAME" --setting "num_shards=$NUM_GPUS"
  --setting "base_port=$BASE_PORT"
  --setting "workers=$WORKERS" --setting "max_tokens=$MAX_TOKENS"
  --setting "temperature=$TEMPERATURE" --setting "top_p=$TOP_P" --setting "seed=$SEED"
  --file "inference_code=$ROOT/scripts/inference.py"
  --file "merge_code=$ROOT/scripts/merge_shards.py"
  --file "preflight=$ROOT/scripts/preflight.py"
  --file "run_metadata=$ROOT/scripts/run_metadata.py"
  --file "requirements=$ROOT/env/requirements.txt"
)
if (( RESUME )); then METADATA_ARGS+=(--verify); fi
"$PYTHON" "$ROOT/scripts/run_metadata.py" "${METADATA_ARGS[@]}"

SHARDS=()
for ((i=0; i<NUM_GPUS; i++)); do SHARDS+=("$OUT_DIR/shard_$i.jsonl"); done
for shard in "${SHARDS[@]}"; do
  [[ "$shard" != "$FINAL" && "$shard" != "$QUESTIONS" && "$shard" != "$ASR" && "$shard" != "$SYSTEM_PROMPT" ]] || {
    echo "shard output path conflicts with an input or final output: $shard" >&2; exit 2;
  }
done
if (( ! RESUME )); then rm -f "${SHARDS[@]}"; fi

echo "[$(date -Is)] starting $NUM_GPUS inference shards"
PIDS=()
for ((i=0; i<NUM_GPUS; i++)); do
  RESUME_ARGS=()
  if (( RESUME )); then RESUME_ARGS+=(--resume); fi
  "$PYTHON" "$ROOT/scripts/inference.py" \
    --base-url "http://127.0.0.1:$((BASE_PORT+i))/v1" \
    --model "$MODEL_NAME" --videos "$VIDEOS" --system-prompt "$SYSTEM_PROMPT" \
    --questions "$QUESTIONS" --asr "$ASR" --output "${SHARDS[$i]}" \
    --shard "$i" --num-shards "$NUM_GPUS" --workers "$WORKERS" \
    --max-tokens "$MAX_TOKENS" --temperature "$TEMPERATURE" --top-p "$TOP_P" --seed "$SEED" \
    "${RESUME_ARGS[@]}" >"$OUT_DIR/shard_$i.log" 2>&1 &
  PIDS+=("$!")
done

FAILED=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=$((FAILED + 1)); done
if (( FAILED > 0 )); then echo "[$(date -Is)] $FAILED inference shards failed; inspect $OUT_DIR" >&2; exit 1; fi

echo "[$(date -Is)] merging $NUM_GPUS shards"
ARGS=(--questions "$QUESTIONS" --output "$FINAL")
for shard in "${SHARDS[@]}"; do ARGS+=(--shard "$shard"); done
"$PYTHON" "$ROOT/scripts/merge_shards.py" "${ARGS[@]}"
"$PYTHON" "$ROOT/scripts/write_manifest.py" \
  --run-metadata "$RUN_METADATA" --output "$FINAL" --manifest "$MANIFEST"

echo "[$(date -Is)] done: $FINAL ($(wc -l < "$FINAL") rows)"
echo "[$(date -Is)] manifest: $MANIFEST"
