# AdInsight-RL

[English](#english) | [中文](#中文)

---

<a id="english"></a>

## English

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

---

<a id="中文"></a>

## 中文

AdInsight-RL 是一个用于广告视频分析的多模态大语言模型。它能够观看广告视频、阅读语音转录文本，并提取营销人员用来打动消费者的核心卖点——涵盖产品特性、客户价值、消费者痛点、实用价值和营销逻辑。

该模型基于 Qwen3.5-9B，通过监督微调（SFT）和组序列策略优化（GSPO）进行适配。本仓库包含部署模型并在官方测试集（3,108 个广告视频）上复现推理结果所需的组件。

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

### 方法概述

AdInsight-RL 使用 Qwen3.5-9B（9,496.37M 参数），冻结视觉编码器与多模态对齐器，仅训练 86.56M LoRA 参数。本推理仓库不重新分发训练数据。

**证据校准数据。** 系统将视频设为主要事实来源，将时间戳 ASR 明确标记为辅助信息。SFT 标签通过多教师蒸馏、独立视频复核和选择性风险审核获得；RL 参考答案表示为 2–4 个带证据模态与时间位置的原子声明，从数据层抑制泛化营销推断及无依据的数字、价格和促销描述。

该数据流程具有三个区别于普通教师过滤的设计。第一，ASR、生成式元数据和音乐描述具有不同的证据权限：视频是权威证据，ASR 只能辅助口播内容，音乐只能支持氛围而不能证明产品事实；音乐分析前还会屏蔽 ASR 语音区间，避免口播产品信息污染声学证据。第二，验证采用风险自适应策略而不是统一重复审核：可靠低风险样本提前停止，数字、医疗、促销和争议声明进入额外审查。第三，RL 参考答案必须经过盲式声明盘点、逐声明审计、保守裁决、对抗挑战和完整性检查，教师模型之间达成一致并不足以直接通过。

**指标对齐提示词。** 训练和推理提示词要求每个编号点仅包含一个声明，只回答问题要求的维度，并删除无证据细节。生成、解析、奖励和评测均以原子声明为统一单位。

**SFT 与 GSPO。** SFT 使用 9,500 条精选视频问答数据，LoRA rank 32 / alpha 64，训练 3 轮，学习率 `2e-5`；使用 8× NVIDIA H20 96GB，耗时约 40 分钟。GSPO 使用 2,902 条训练数据和 70 条验证数据，每个问题采样 8 个回答，采用序列级重要性采样、组级奖励缩放和连续软匹配声明奖励；使用 7× H20，耗时约 7 小时 55 分钟，最终根据验证奖励选择 step 300。

| 阶段 | 数据 | 可训练参数 | 硬件 | 训练时长 | 最佳检查点 |
|---|---:|---:|---:|---:|---:|
| SFT | 9,500 | 86.56M（0.91%） | 8× H20 96GB | 约 40 分钟 | step 57 |
| GSPO | 2,902（+70 验证） | 86.56M（0.91%） | 7× H20 96GB | 约 7 小时 55 分钟 | step 300 |

**奖励与推理。** 奖励函数组合软匹配声明级 F1、Precision、Recall、重复惩罚和无依据扩张惩罚（见 `training/reward_plugin.py`）。推理阶段将视频与带可靠性标记的 ASR 输入 vLLM；每张 GPU 部署一个 TP=1 实例、最多并发 4 个序列，客户端质量门约束输出为 2–4 个连续编号的原子声明。
