# AdInsight-RL

[English](README.md) | [简体中文](README.zh-CN.md)

---

AdInsight-RL is a multimodal large language model designed for advertising video analysis. It watches ad videos, reads the speech transcript, and extracts the core selling points that marketers use to persuade consumers — covering product features, customer value, consumer pain points, practical benefits, and marketing logic.

The model is adapted from Qwen3.5-9B through supervised fine-tuning (SFT) and Group Sequence Policy Optimization (GSPO). This repository contains the components needed to deploy the model and reproduce inference on the official test set of 3,108 advertising videos.

### Reproduction Guide

#### Step 1: Install

```bash
pip install -r env/requirements.txt
```

Requires: Python ≥ 3.10, CUDA ≥ 12.4, NVIDIA driver ≥ 535, at least 1 GPU with ≥ 24GB VRAM.

#### Step 2: Download the models

Download the main model and the ASR model from HuggingFace:

- **Main model (LLM)**: [Harris15767946495/AdInsight-RL](https://huggingface.co/Harris15767946495/AdInsight-RL)
- **ASR model (Whisper)**: [deepdml/faster-whisper-large-v3-turbo-ct2](https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2)

Then set the environment variables:

```bash
export ADINSIGHT_MODEL=/path/to/AdInsight-RL
export ADINSIGHT_MODEL_NAME=AdInsight-RL-Step300
export ADINSIGHT_VIDEO_DIR=/path/to/mars2_videos
export WHISPER_MODEL=/path/to/faster-whisper-large-v3-turbo-ct2
```

#### Step 3: Deploy the model

```bash
bash scripts/deploy.sh
```

Auto-detects available GPUs and starts one vLLM service per GPU. Override:

```bash
NUM_GPUS=4 bash scripts/deploy.sh   # use only 4 GPUs
```

#### Step 4: Generate ASR (or use the reference)

ASR is a key auxiliary input — the model uses video as primary evidence and ASR transcript as supporting evidence.

**Option A: Generate from scratch** (auto-detects GPU count, supports resume):

```bash
bash scripts/run_asr.sh
# → data/asr_final.jsonl (3,108 rows)
```

**Option B: Use the provided reference ASR**:

```bash
cp data/asr_reference.jsonl data/asr_final.jsonl
```

#### Step 5: Run inference

```bash
bash scripts/run_inference.sh
# → outputs/submission.jsonl (3,108 rows, official format)
```

Auto-detects GPU count, runs N shards in parallel, merges into official format: `{"id":"...","model_prediction":"1. ..."}`. Supports `--resume` per shard.

### Project Structure

```
AdInsight-RL/
├── scripts/
│   ├── deploy.sh              # ① deploy model with vLLM (auto GPU count)
│   ├── run_asr.sh             # ② ASR generation (auto GPU count, resume)
│   ├── generate_asr.py        #   faster-whisper transcription (with --resume)
│   ├── postprocess_asr.py     #   merge segments, fix EOF, reliability
│   ├── run_inference.sh       # ③ parallel inference + merge (auto GPU count)
│   ├── inference.py           #   video + ASR → model → answer (with --resume)
│   └── merge_shards.py        #   merge → official submission format
├── training/
│   └── reward_plugin.py       # RL reward function (reference)
├── prompts/
│   └── final_answer.txt       # system prompt
├── data/
│   ├── MAC_QA.jsonl           # official test set (3,108 questions)
│   └── asr_reference.jsonl    # frozen ASR reference (3,108 rows)
└── env/requirements.txt
```

### Method Overview

AdInsight-RL uses Qwen3.5-9B (9,496.37M parameters) and trains 86.56M LoRA parameters while freezing the visual encoder and multimodal aligner. Training data is not redistributed in this inference package.

**Evidence-calibrated data.** Video is the primary source of truth; timestamped ASR is an explicitly labeled auxiliary cue. SFT targets are produced by multi-teacher distillation, video verification, and selective risk review. RL references are represented as 2–4 atomic claims with evidence modality and temporal provenance. This construction discourages generic marketing inferences and unsupported numerical or promotional claims.

Three details distinguish the data pipeline from ordinary teacher filtering. First, ASR, generated metadata, and music descriptions are assigned different evidence permissions: video is authoritative, ASR is only a speech cue, and music can support atmosphere but never product facts. Speech intervals are masked before music analysis to prevent spoken product claims from contaminating acoustic evidence. Second, verification is risk-adaptive rather than uniformly repeated: reliable low-risk samples stop early, while numerical, medical, promotional, and disputed claims receive additional review. Third, RL references must pass blind inventory, claim-level audit, conservative adjudication, adversarial challenge, and completeness checking; teacher agreement alone is never sufficient.

**Metric-aligned prompting.** Training and inference prompts require one claim per numbered point, answer only the dimensions requested by the question, and omit unsupported details. The same atomic-claim unit is used by generation, parsing, reward computation, and evaluation.

**SFT and GSPO.** SFT uses 9,500 curated video-QA pairs, LoRA rank 32 / alpha 64, three epochs, and learning rate `2e-5`; it takes approximately 40 minutes on 8× NVIDIA H20 96GB GPUs. GSPO uses 2,902 training and 70 development prompts, eight generations per prompt, sequence-level importance sampling, group reward scaling, and a continuous soft-matching claim reward. It takes approximately 7 h 55 min on 7× H20 GPUs; validation selects step 300.

| Stage | Data | Trainable parameters | Hardware | Duration | Selection |
|---|---:|---:|---:|---:|---:|
| SFT | 9,500 | 86.56M (0.91%) | 8× H20 96GB | ~40 min | step 57 |
| GSPO | 2,902 (+70 dev) | 86.56M (0.91%) | 7× H20 96GB | ~7 h 55 min | step 300 |

**Reward and inference.** The reward combines soft claim-level F1, precision, recall, duplicate penalties, and unsupported-overclaim penalties (see `training/reward_plugin.py`). Inference supplies video and reliability-labeled ASR to vLLM. Deployment uses one tensor-parallel-1 instance per GPU with up to four concurrent sequences; client-side validation enforces 2–4 continuously numbered claims.
