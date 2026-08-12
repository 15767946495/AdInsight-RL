#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_asr.sh --videos DIR [options]

Options:
  --videos DIR           Directory containing <question-id>.mp4 files
  --questions FILE       Expected question IDs (default: data/MAC_QA.jsonl)
  --model MODEL          Whisper model ID or local CT2 directory
  --model-revision REV   Frozen Hugging Face model revision
  --download-root DIR    Whisper model download/cache directory
  --raw-dir DIR          Raw shard directory (default: data/asr_raw)
  --output FILE          Final ASR JSONL (default: data/asr_final.jsonl)
  --num-gpus N           Number of ASR shards (default: all visible GPUs)
  --gpu-ids LIST         Comma-separated physical GPU IDs (default: CUDA_VISIBLE_DEVICES)
  --device DEVICE        faster-whisper device (default: cuda)
  --compute-type TYPE    faster-whisper compute type (default: float16)
  --beam-size N          Beam size (default: 5)
  --merge-gap SECONDS    Segment merge gap (default: 0.5)
  --max-chars N          Maximum transcript characters (default: 12000)
  --resume               Resume successful records in current shard files
  --no-resume            Start fresh (default)
  --python PATH          Python executable (default: python)
  -h, --help             Show this help
EOF
}

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON_BIN:-python}"
VIDEOS="${ADINSIGHT_VIDEO_DIR:-}"
QUESTIONS="$ROOT/data/MAC_QA.jsonl"
WHISPER_MODEL="${WHISPER_MODEL:-large-v3-turbo}"
MODEL_REVISION=""
DOWNLOAD_ROOT=""
RAW_DIR="$ROOT/data/asr_raw"
FINAL="$ROOT/data/asr_final.jsonl"
NUM_GPUS="${NUM_GPUS:-}"
GPU_IDS_RAW="${CUDA_VISIBLE_DEVICES:-}"
DEVICE="cuda"
COMPUTE_TYPE="float16"
BEAM_SIZE=5
MERGE_GAP=0.5
MAX_CHARS=12000
RESUME=0

while (( $# )); do
  case "$1" in
    --videos) VIDEOS="${2:?Missing value for --videos}"; shift 2 ;;
    --questions) QUESTIONS="${2:?Missing value for --questions}"; shift 2 ;;
    --model) WHISPER_MODEL="${2:?Missing value for --model}"; shift 2 ;;
    --model-revision) MODEL_REVISION="${2:?Missing value for --model-revision}"; shift 2 ;;
    --download-root) DOWNLOAD_ROOT="${2:?Missing value for --download-root}"; shift 2 ;;
    --raw-dir) RAW_DIR="${2:?Missing value for --raw-dir}"; shift 2 ;;
    --output) FINAL="${2:?Missing value for --output}"; shift 2 ;;
    --num-gpus) NUM_GPUS="${2:?Missing value for --num-gpus}"; shift 2 ;;
    --gpu-ids) GPU_IDS_RAW="${2:?Missing value for --gpu-ids}"; shift 2 ;;
    --device) DEVICE="${2:?Missing value for --device}"; shift 2 ;;
    --compute-type) COMPUTE_TYPE="${2:?Missing value for --compute-type}"; shift 2 ;;
    --beam-size) BEAM_SIZE="${2:?Missing value for --beam-size}"; shift 2 ;;
    --merge-gap) MERGE_GAP="${2:?Missing value for --merge-gap}"; shift 2 ;;
    --max-chars) MAX_CHARS="${2:?Missing value for --max-chars}"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --no-resume) RESUME=0; shift ;;
    --python) PYTHON="${2:?Missing value for --python}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$VIDEOS" ]] || { echo "--videos is required" >&2; exit 2; }
[[ -d "$VIDEOS" ]] || { echo "Video directory not found: $VIDEOS" >&2; exit 2; }
[[ -f "$QUESTIONS" ]] || { echo "Questions file not found: $QUESTIONS" >&2; exit 2; }
VIDEOS="$(cd "$VIDEOS" && pwd)"
QUESTIONS="$(cd "$(dirname "$QUESTIONS")" && pwd)/$(basename "$QUESTIONS")"
if [[ -d "$WHISPER_MODEL" ]]; then WHISPER_MODEL="$(cd "$WHISPER_MODEL" && pwd)"; fi
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
[[ "$MAX_CHARS" =~ ^[1-9][0-9]*$ ]] && (( MAX_CHARS <= 12000 )) || {
  echo "--max-chars must be an integer between 1 and 12000" >&2; exit 2;
}
command -v "$PYTHON" >/dev/null || { echo "Python executable not found: $PYTHON" >&2; exit 2; }
mkdir -p "$RAW_DIR" "$(dirname "$FINAL")"
if [[ -n "$DOWNLOAD_ROOT" ]]; then mkdir -p "$DOWNLOAD_ROOT"; DOWNLOAD_ROOT="$(cd "$DOWNLOAD_ROOT" && pwd)"; fi
RAW_DIR="$(cd "$RAW_DIR" && pwd)"
FINAL="$(cd "$(dirname "$FINAL")" && pwd)/$(basename "$FINAL")"
[[ "$FINAL" != "$QUESTIONS" ]] || { echo "--output must not overwrite the questions file" >&2; exit 2; }

"$PYTHON" "$ROOT/scripts/preflight.py" videos --questions "$QUESTIONS" --videos "$VIDEOS"

RUN_METADATA="$RAW_DIR/run_metadata.json"
METADATA_ARGS=(
  --kind asr --questions "$QUESTIONS" --videos "$VIDEOS"
  --model "$WHISPER_MODEL" --model-revision "$MODEL_REVISION" --output "$RUN_METADATA"
  --setting "device=$DEVICE" --setting "compute_type=$COMPUTE_TYPE"
  --setting "beam_size=$BEAM_SIZE" --setting "sample_rate=16000"
  --setting "merge_gap=$MERGE_GAP" --setting "max_chars=$MAX_CHARS"
  --file "generate_asr=$ROOT/scripts/generate_asr.py"
  --file "postprocess_asr=$ROOT/scripts/postprocess_asr.py"
  --file "preflight=$ROOT/scripts/preflight.py"
  --file "run_metadata=$ROOT/scripts/run_metadata.py"
  --file "requirements=$ROOT/env/requirements.txt"
)
if (( RESUME )); then METADATA_ARGS+=(--verify); fi
"$PYTHON" "$ROOT/scripts/run_metadata.py" "${METADATA_ARGS[@]}"

SHARDS=()
for ((i=0; i<NUM_GPUS; i++)); do SHARDS+=("$RAW_DIR/shard_$i.jsonl"); done
if (( ! RESUME )); then rm -f "${SHARDS[@]}"; fi

echo "[$(date -Is)] transcribing on $NUM_GPUS GPU(s) with faster-whisper"
PIDS=()
for ((i=0; i<NUM_GPUS; i++)); do
  RESUME_ARGS=()
  if (( RESUME )); then RESUME_ARGS+=(--resume); fi
  DOWNLOAD_ARGS=()
  if [[ -n "$DOWNLOAD_ROOT" ]]; then DOWNLOAD_ARGS+=(--download-root "$DOWNLOAD_ROOT"); fi
  if [[ -n "$MODEL_REVISION" ]]; then DOWNLOAD_ARGS+=(--model-revision "$MODEL_REVISION"); fi
  CUDA_VISIBLE_DEVICES="${GPU_LIST[$i]}" "$PYTHON" "$ROOT/scripts/generate_asr.py" \
    --videos "$VIDEOS" --output "${SHARDS[$i]}" \
    --model "$WHISPER_MODEL" --device "$DEVICE" --compute-type "$COMPUTE_TYPE" \
    --beam-size "$BEAM_SIZE" \
    --shard "$i" --num-shards "$NUM_GPUS" "${DOWNLOAD_ARGS[@]}" "${RESUME_ARGS[@]}" \
    >"$RAW_DIR/shard_$i.log" 2>&1 &
  PIDS+=("$!")
done

FAILED=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=$((FAILED + 1)); done
if (( FAILED > 0 )); then echo "[$(date -Is)] $FAILED ASR shards failed; inspect $RAW_DIR" >&2; exit 1; fi

echo "[$(date -Is)] merging and post-processing"
MERGED="$RAW_DIR/merged.jsonl"
: > "$MERGED"
for shard in "${SHARDS[@]}"; do cat "$shard" >> "$MERGED"; done
"$PYTHON" "$ROOT/scripts/postprocess_asr.py" \
  --input "$MERGED" --videos "$VIDEOS" --questions "$QUESTIONS" --output "$FINAL" \
  --merge-gap "$MERGE_GAP" --max-chars "$MAX_CHARS"
"$PYTHON" "$ROOT/scripts/preflight.py" inference --questions "$QUESTIONS" --videos "$VIDEOS" --asr "$FINAL"

echo "[$(date -Is)] done: $FINAL ($(wc -l < "$FINAL") rows)"
