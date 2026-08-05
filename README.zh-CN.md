# AdInsight-RL

[English](README.md) | [简体中文](README.zh-CN.md)

---

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
