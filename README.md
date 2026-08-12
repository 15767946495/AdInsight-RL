# AdInsight-RL

[English](README.md) | [简体中文](README.zh-CN.md)

AdInsight-RL is a multimodal model for advertising video analysis. It is adapted from Qwen3.5-9B with SFT and GSPO and extracts selling points, customer value, consumer pain points, practical benefits, and marketing logic from videos with auxiliary ASR evidence.

This repository reproduces **all 3,108 final inference results from the official test videos and the released final weights**. It does not reproduce SFT or GSPO training.

## Frozen Resources

Use these immutable revisions to prevent upstream updates from changing results:

| Resource | Repository | Revision |
|---|---|---|
| AdInsight-RL | [Harris15767946495/AdInsight-RL](https://huggingface.co/Harris15767946495/AdInsight-RL) | `f386fe576b551a6345b1a96a014ab7248bfb40a8` |
| faster-whisper | [deepdml/faster-whisper-large-v3-turbo-ct2](https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2) | `4df90f75321148c3a29a9e2351b7ddf8f5b115a8` |

Frozen repository data hashes:

```text
abe741d997ac467f1444255d7123228603a942cd7843bd364bb6f1fef3044e9a  data/MAC_QA.jsonl
83d7f0cc4f1b8dc939a8b38840d05caa043fe9e29d511bce111c08983c10188a  data/asr_reference.jsonl
```

## Requirements

- Python 3.10–3.13
- CUDA >= 12.4 and NVIDIA driver >= 535
- At least one NVIDIA GPU with >= 24 GB VRAM
- `ffmpeg`, `curl`, and `nvidia-smi`

Install from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r env/requirements.txt
```

## Data Layout

The official test videos are not redistributed by this repository. Put all 3,108 videos in one directory, named `<question-id>.mp4`:

```text
videos/
├── 1d9c3227-2002-4aae-b6c1-57c44c2ef914.mp4
├── 4325fcae-87c4-420c-a1e7-83a27a21c75d.mp4
└── ...
```

The examples below use configurable relative paths:

```bash
VIDEO_DIR=../mars2_videos
WORK_DIR=../adinsight_work
MODEL_DIR="$WORK_DIR/models/AdInsight-RL"
ASR_MODEL_DIR="$WORK_DIR/models/faster-whisper-large-v3-turbo-ct2"
mkdir -p "$WORK_DIR/models" "$WORK_DIR/asr" "$WORK_DIR/outputs" "$WORK_DIR/logs"
```

Every data, model, log, and output path is accepted through CLI arguments. Run any script with `--help` for all options.

## Step 1: Download Frozen Weights

```bash
hf download Harris15767946495/AdInsight-RL \
  --revision f386fe576b551a6345b1a96a014ab7248bfb40a8 \
  --local-dir "$MODEL_DIR"

hf download deepdml/faster-whisper-large-v3-turbo-ct2 \
  --revision 4df90f75321148c3a29a9e2351b7ddf8f5b115a8 \
  --local-dir "$ASR_MODEL_DIR"
```

## Step 2: Prepare ASR

### Option A: Generate from videos

Run ASR before deploying vLLM so the models do not compete for GPU memory.

```bash
bash scripts/run_asr.sh \
  --videos "$VIDEO_DIR" \
  --questions data/MAC_QA.jsonl \
  --model "$ASR_MODEL_DIR" \
  --model-revision 4df90f75321148c3a29a9e2351b7ddf8f5b115a8 \
  --raw-dir "$WORK_DIR/asr/raw" \
  --output "$WORK_DIR/asr/asr_final.jsonl" \
  --num-gpus 4
```

Any transcription failure returns a nonzero exit code. After correcting the cause, add `--resume` to reuse successful records only.

### Option B: Use frozen reference ASR

```bash
ASR_FILE=data/asr_reference.jsonl
python scripts/preflight.py inference \
  --questions data/MAC_QA.jsonl \
  --videos "$VIDEO_DIR" \
  --asr "$ASR_FILE"
```

For option A, set:

```bash
ASR_FILE="$WORK_DIR/asr/asr_final.jsonl"
```

## Step 3: Deploy

```bash
bash scripts/deploy.sh \
  --model "$MODEL_DIR" \
  --model-name AdInsight-RL-Step300 \
  --model-revision f386fe576b551a6345b1a96a014ab7248bfb40a8 \
  --video-dir "$VIDEO_DIR" \
  --log-dir "$WORK_DIR/logs" \
  --base-port 8240 \
  --num-gpus 4
```

The script starts one TP=1 vLLM service per GPU and returns only after every service passes its health check.

## Step 4: Infer and Write Verification Manifest

```bash
bash scripts/run_inference.sh \
  --questions data/MAC_QA.jsonl \
  --asr "$ASR_FILE" \
  --videos "$VIDEO_DIR" \
  --output-dir "$WORK_DIR/outputs" \
  --output results/submission.jsonl \
  --manifest results/submission.manifest.json \
  --model-name AdInsight-RL-Step300 \
  --model-source "$MODEL_DIR" \
  --model-revision f386fe576b551a6345b1a96a014ab7248bfb40a8 \
  --base-port 8240 \
  --num-gpus 4
```

Before inference, the pipeline verifies complete coverage of all 3,108 questions, videos, and ASR records. Final merging succeeds only when:

- there are no missing, duplicate, or extra IDs;
- no prediction is empty or contains `ERROR`;
- every answer has 2–4 continuously numbered points;
- output order matches `MAC_QA.jsonl`.

`submission.manifest.json` records the model revision, SHA256 hashes of every input, prompt, and output, and all inference parameters. It does not record machine-specific absolute paths.

## Final Review Package

The review link must contain at least:

- this complete code repository;
- complete weights at the frozen revisions above;
- `data/MAC_QA.jsonl` and `data/asr_reference.jsonl`;
- `results/submission.jsonl`;
- `results/submission.manifest.json`;
- instructions and licensing information for obtaining the official test videos.

The repository does not fabricate a placeholder submission. Before publishing the final review link, run the real released weights and commit both generated files under `results/`.

## Project Structure

```text
AdInsight-RL/
├── scripts/
│   ├── deploy.sh              # CLI-configured vLLM deployment
│   ├── run_asr.sh             # sharded ASR generation, merge, validation
│   ├── generate_asr.py        # faster-whisper transcription
│   ├── postprocess_asr.py     # ASR post-processing and coverage checks
│   ├── preflight.py           # question, video, and ASR validation
│   ├── run_inference.sh       # parallel inference and exact shard merge
│   ├── inference.py           # video + ASR inference client
│   ├── merge_shards.py        # final format and coverage validation
│   ├── run_metadata.py        # input/model hashes and resume fingerprint
│   └── write_manifest.py      # verifiable result manifest
├── prompts/final_answer.txt
├── data/MAC_QA.jsonl
├── data/asr_reference.jsonl
├── results/                   # final submission and manifest
└── env/requirements.txt
```

## Method Overview

AdInsight-RL uses Qwen3.5-9B and trains 86.56M LoRA parameters while freezing the visual encoder and multimodal aligner. SFT uses 9,500 curated video-QA pairs with LoRA rank 32 / alpha 64 for three epochs at `2e-5`. GSPO uses 2,902 training and 70 validation prompts with eight generations per prompt; validation selects step 300.

Training data construction uses multi-teacher distillation, independent video verification, risk review, speech-masked music analysis, and atomic-claim audits. `training/reward_plugin.py` provides the soft claim-matching reward as a method reference, but the training data and full retraining pipeline are outside this inference reproduction package.
