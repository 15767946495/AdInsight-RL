# AdInsight-RL

[English](#english) | [中文](#中文)

---

<a id="english"></a>

## English

AdInsight-RL is a multimodal large language model designed for advertising video analysis. It watches ad videos, reads the speech transcript, and extracts the core selling points that marketers use to persuade consumers — covering product features, customer value, consumer pain points, practical benefits, and marketing logic.

The model is fine-tuned from Qwen3.5-9B through a two-stage pipeline: supervised fine-tuning (SFT) followed by reinforcement learning (GRPO). This repository contains everything needed to deploy the model and reproduce the inference results on the official test set of 3,108 advertising videos.

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

### Training Process

The model was trained in two stages (training data is not included in this package):

**Stage 1: SFT (Supervised Fine-Tuning)**
- Base: Qwen3.5-9B (official post-trained checkpoint)
- Data: 9,500 curated video QA pairs (teacher-generated, judge-filtered, risk-reviewed)
- Method: LoRA (rank 32, alpha 64, all-linear), 3 epochs, lr 2e-5, freeze vision tower + aligner
- Best checkpoint selected by eval_loss

**Stage 2: RL (Reinforcement Learning — GRPO with sequence-level importance sampling)**
- Base: SFT merged model
- Data: 2,902 train / 70 dev samples with atomic reference points (video + question + ASR → reference claims)
- Reward: claim-level F1 with soft matching + precision + recall components (see `training/reward_plugin.py`)
- Method: LoRA (rank 32, alpha 64), 1 epoch (414 steps), lr 5e-7, temperature 1.1, beta 0.04
- 8 generations per prompt, group reward scaling, DeepSpeed ZeRO-2, vLLM colocate

The final model is the step-300 LoRA adapter merged with the base model via `swift export --merge_lora true`.

---

<a id="中文"></a>

## 中文

AdInsight-RL 是一个用于广告视频分析的多模态大语言模型。它能够观看广告视频、阅读语音转录文本，并提取营销人员用来打动消费者的核心卖点——涵盖产品特性、客户价值、消费者痛点、实用价值和营销逻辑。

该模型基于 Qwen3.5-9B，通过两阶段流程微调而成：监督微调（SFT）+ 强化学习（GRPO）。本仓库包含部署模型并在官方测试集（3,108 个广告视频）上复现推理结果所需的全部内容。

### 复现指南

#### 步骤 1：安装依赖

```bash
pip install -r env/requirements.txt
```

环境要求：Python ≥ 3.10，CUDA ≥ 12.4，NVIDIA 驱动 ≥ 535，至少 1 张显存 ≥ 24GB 的 GPU。

#### 步骤 2：下载模型

从 HuggingFace 下载主模型和 ASR 模型：

- **主模型（LLM）**：[Harris15767946495/AdInsight-RL](https://huggingface.co/Harris15767946495/AdInsight-RL)
- **ASR 模型（Whisper）**：[deepdml/faster-whisper-large-v3-turbo-ct2](https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2)

然后设置环境变量：

```bash
export ADINSIGHT_MODEL=/path/to/AdInsight-RL
export ADINSIGHT_MODEL_NAME=AdInsight-RL-Step300
export ADINSIGHT_VIDEO_DIR=/path/to/mars2_videos
export WHISPER_MODEL=/path/to/faster-whisper-large-v3-turbo-ct2
```

#### 步骤 3：部署模型

```bash
bash scripts/deploy.sh
```

自动检测可用 GPU 数量，每个 GPU 启动一个 vLLM 服务。可手动指定：

```bash
NUM_GPUS=4 bash scripts/deploy.sh   # 只使用 4 张 GPU
```

#### 步骤 4：生成 ASR（或使用参考数据）

ASR 是重要的辅助输入——模型以视频为主要证据，ASR 转录文本为支撑证据。

**方案 A：从头生成**（自动检测 GPU 数量，支持断点续传）：

```bash
bash scripts/run_asr.sh
# → data/asr_final.jsonl（3,108 行）
```

**方案 B：使用提供的参考 ASR**：

```bash
cp data/asr_reference.jsonl data/asr_final.jsonl
```

#### 步骤 5：运行推理

```bash
bash scripts/run_inference.sh
# → outputs/submission.jsonl（3,108 行，官方格式）
```

自动检测 GPU 数量，并行运行 N 个分片，合并为官方提交格式：`{"id":"...","model_prediction":"1. ..."}`。每个分片支持 `--resume` 断点续传。

### 项目结构

```
AdInsight-RL/
├── scripts/
│   ├── deploy.sh              # ① 使用 vLLM 部署模型（自动检测 GPU 数量）
│   ├── run_asr.sh             # ② ASR 生成（自动检测 GPU 数量，支持断点续传）
│   ├── generate_asr.py        #   faster-whisper 转录（支持 --resume）
│   ├── postprocess_asr.py     #   合并片段、修正 EOF、评估可靠性
│   ├── run_inference.sh       # ③ 并行推理 + 合并（自动检测 GPU 数量）
│   ├── inference.py           #   视频 + ASR → 模型 → 答案（支持 --resume）
│   └── merge_shards.py        #   合并为官方提交格式
├── training/
│   └── reward_plugin.py       # RL 奖励函数（参考）
├── prompts/
│   └── final_answer.txt       # 系统提示词
├── data/
│   ├── MAC_QA.jsonl           # 官方测试集（3,108 个问题）
│   └── asr_reference.jsonl    # 冻结的 ASR 参考数据（3,108 行）
└── env/requirements.txt
```

### 训练过程

模型分两个阶段训练（训练数据未包含在此包中）：

**阶段 1：SFT（监督微调）**
- 基座：Qwen3.5-9B（官方后训练检查点）
- 数据：9,500 条精选视频问答对（教师模型生成、裁判过滤、风险审核）
- 方法：LoRA（rank 32，alpha 64，所有线性层），3 轮，lr 2e-5，冻结视觉编码器 + 对齐层
- 根据 eval_loss 选择最佳检查点

**阶段 2：RL（强化学习 — GRPO，序列级重要性采样）**
- 基座：SFT 合并后的模型
- 数据：2,902 训练 / 70 验证样本，含原子级参考要点（视频 + 问题 + ASR → 参考声明）
- 奖励：声明级 F1，软匹配 + 精确率 + 召回率组合（见 `training/reward_plugin.py`）
- 方法：LoRA（rank 32，alpha 64），1 轮（414 步），lr 5e-7，temperature 1.1，beta 0.04
- 每个 prompt 生成 8 个回复，组内奖励缩放，DeepSpeed ZeRO-2，vLLM 共置

最终模型是 step-300 的 LoRA 适配器与基座模型合并后的版本，通过 `swift export --merge_lora true` 导出。
