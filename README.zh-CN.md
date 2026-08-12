# AdInsight-RL

[English](README.md) | [简体中文](README.zh-CN.md)

AdInsight-RL 是一个用于广告视频分析的多模态大语言模型。它基于 Qwen3.5-9B，经 SFT 和 GSPO 适配，从广告视频及辅助 ASR 中提取核心卖点、客户价值、消费者痛点、实用价值和营销逻辑。

本仓库的复现范围是：**从官方测试视频和已发布的最终权重开始，生成完整的 3,108 条最终推理结果**。不包含 SFT/GSPO 重训练。

## 冻结资源

为避免上游仓库更新导致结果漂移，请使用以下 revision：

| 资源 | 仓库 | Revision |
|---|---|---|
| AdInsight-RL | [Harris15767946495/AdInsight-RL](https://huggingface.co/Harris15767946495/AdInsight-RL) | `f386fe576b551a6345b1a96a014ab7248bfb40a8` |
| faster-whisper | [deepdml/faster-whisper-large-v3-turbo-ct2](https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2) | `4df90f75321148c3a29a9e2351b7ddf8f5b115a8` |

仓库内冻结数据校验值：

```text
abe741d997ac467f1444255d7123228603a942cd7843bd364bb6f1fef3044e9a  data/MAC_QA.jsonl
83d7f0cc4f1b8dc939a8b38840d05caa043fe9e29d511bce111c08983c10188a  data/asr_reference.jsonl
```

## 环境要求

- Python 3.10–3.13
- CUDA >= 12.4，NVIDIA 驱动 >= 535
- 至少 1 张显存 >= 24 GB 的 NVIDIA GPU
- `ffmpeg`、`curl` 和 `nvidia-smi`

从仓库根目录安装：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r env/requirements.txt
```

## 数据布局

官方测试视频不由本仓库重新分发。将全部 3,108 个视频放在同一目录，文件名必须是 `<question-id>.mp4`：

```text
videos/
├── 1d9c3227-2002-4aae-b6c1-57c44c2ef914.mp4
├── 4325fcae-87c4-420c-a1e7-83a27a21c75d.mp4
└── ...
```

后续示例使用可自行修改的相对路径：

```bash
VIDEO_DIR=../mars2_videos
WORK_DIR=../adinsight_work
MODEL_DIR="$WORK_DIR/models/AdInsight-RL"
ASR_MODEL_DIR="$WORK_DIR/models/faster-whisper-large-v3-turbo-ct2"
mkdir -p "$WORK_DIR/models" "$WORK_DIR/asr" "$WORK_DIR/outputs" "$WORK_DIR/logs"
```

所有数据路径、模型路径、日志目录和输出目录都由 CLI 参数传入。各脚本可通过 `--help` 查看完整参数。

## 步骤 1：下载固定权重

```bash
hf download Harris15767946495/AdInsight-RL \
  --revision f386fe576b551a6345b1a96a014ab7248bfb40a8 \
  --local-dir "$MODEL_DIR"

hf download deepdml/faster-whisper-large-v3-turbo-ct2 \
  --revision 4df90f75321148c3a29a9e2351b7ddf8f5b115a8 \
  --local-dir "$ASR_MODEL_DIR"
```

## 步骤 2：准备 ASR

### 方案 A：从视频重新生成

此步骤必须在部署 vLLM 之前运行，避免两个模型争用 GPU 显存。

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

失败的转录会导致命令返回非零；修复原因后加入 `--resume`，只复用成功记录。

### 方案 B：使用冻结参考 ASR

```bash
ASR_FILE=data/asr_reference.jsonl
python scripts/preflight.py inference \
  --questions data/MAC_QA.jsonl \
  --videos "$VIDEO_DIR" \
  --asr "$ASR_FILE"
```

若使用方案 A：

```bash
ASR_FILE="$WORK_DIR/asr/asr_final.jsonl"
```

## 步骤 3：部署模型

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

脚本每张 GPU 启动一个 TP=1 的 vLLM 服务，并在全部服务通过健康检查后返回。

## 步骤 4：运行推理并生成验证清单

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

推理前会检查 3,108 个问题、视频和 ASR 的完整覆盖。最终合并仅在以下条件全部满足时成功：

- ID 无缺失、重复或额外项；
- 每条结果非空且不包含 `ERROR`；
- 每条结果包含 2–4 个连续编号点；
- 最终输出顺序与 `MAC_QA.jsonl` 一致。

`submission.manifest.json` 记录模型 revision、输入/提示词/输出 SHA256 和全部推理参数，不记录本机绝对路径。

## 最终审查包

发布给审查方的链接必须至少包含：

- 本仓库完整代码；
- 上述冻结 revision 的完整模型权重；
- `data/MAC_QA.jsonl` 和 `data/asr_reference.jsonl`；
- `results/submission.jsonl`；
- `results/submission.manifest.json`；
- 官方测试视频的获取方式和许可说明。

仓库目前不包含可伪造的占位提交结果。发布前必须用真实权重运行上述命令，并提交 `results/` 下两个实际生成文件。

## 项目结构

```text
AdInsight-RL/
├── scripts/
│   ├── deploy.sh              # CLI 配置的 vLLM 部署
│   ├── run_asr.sh             # ASR 分片生成、合并和校验
│   ├── generate_asr.py        # faster-whisper 转录
│   ├── postprocess_asr.py     # ASR 后处理与覆盖校验
│   ├── preflight.py           # 视频、问题、ASR 完整性预检
│   ├── run_inference.sh       # 并行推理和精确分片合并
│   ├── inference.py           # 视频 + ASR 推理客户端
│   ├── merge_shards.py        # 最终格式和覆盖校验
│   ├── run_metadata.py        # 输入/模型哈希和恢复指纹
│   └── write_manifest.py      # 生成可验证结果清单
├── prompts/final_answer.txt
├── data/MAC_QA.jsonl
├── data/asr_reference.jsonl
├── results/                   # 最终提交文件和 manifest
└── env/requirements.txt
```

## 方法概述

AdInsight-RL 使用 Qwen3.5-9B，冻结视觉编码器与多模态对齐器，仅训练 86.56M LoRA 参数。SFT 使用 9,500 条精选视频问答数据，LoRA rank 32 / alpha 64，训练 3 轮，学习率 `2e-5`。GSPO 使用 2,902 条训练数据和 70 条验证数据，每个问题采样 8 个回答，最终根据验证奖励选择 step 300。

训练数据构建包含多教师蒸馏、独立视频复核、风险审核、语音区间屏蔽后的音乐分析和原子声明审计。`training/reward_plugin.py` 提供声明级软匹配奖励函数作为方法参考，但训练数据和完整重训练流水线不属于本推理复现包。
