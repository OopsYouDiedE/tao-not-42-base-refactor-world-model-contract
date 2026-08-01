# TAO

本项目使用 MineStudio Minecraft 轨迹训练视觉动作模型，并通过 CraftGround 执行 TAP
命名动作协议。仓库只保存源码、测试和稳定契约。数据集、图片、审核结果、日志与模型权重统一写入
`runs/`，该目录不进入版本控制。

## 当前管线

| 阶段 | 输入 | 正式入口 | 输出 |
|---|---|---|---|
| 下载 | Hugging Face MineStudio 数据 | `dataset.extraction.minestudio.download` | LMDB 模态分片 |
| 预训练数据 | LMDB 图像与动作 | `dataset.organization.pretraining` | TAP JSONL 与图片 |
| 流式训练数据 | LMDB 图像与动作 | `train.bc.streaming_dataset` | 内存中的视觉对话样本 |
| 轨迹题生成 | LMDB 轨迹 | `dataset.organization.generate_questions` | 候选题、隔离答案和审核请求 |
| 人工双审 | 候选题与原始轨迹 | `dataset.review.questions`、`dataset.review.actions` | 题目审核与规范动作答案 |
| 训练归档 | 双审通过题目 | `dataset.organization.pack_hdf5` | 自包含 HDF5 |
| 视觉 SFT | LMDB、落盘 JSONL 或 HDF5 | `train.bc.gemma_vision_sft`、`train.bc.qwen_vision_sft` | LoRA adapter |
| 教师基线 | CraftGround 同一快照 | `tools.audits.codex_teacher_batch8` | 前 50% 教师轨迹与 BC JSONL |
| 策略 RLHF | 本地 LoRA 的 on-policy rollout | `train.rlhf.gemma_vision_rlhf` | LoRA adapter |
| 闭环执行 | TAP 动作文本 | `game_environment.closed_loop_server` | 逐 tick RGB 与轨迹 JSON |

`dataset/organization/README.md` 说明轨迹题协议、双审准入条件和 HDF5 训练格式。
`docs/game_environment_reset.md` 说明当前 CraftGround 内存快照边界。
`docs/craftground_closed_loop.md` 说明闭环 HTTP 契约。

## 代码结构

| 路径 | 职责 |
|---|---|
| `tao/protocols/` | TAP 动作协议与执行契约 |
| `tao/baselines/codex/` | Codex CLI 调用、教师生成、统一评分与筛选 |
| `dataset/extraction/` | MineStudio 下载和 LMDB 原始轨迹读取 |
| `dataset/organization/` | 数据划分、样本生成、协议整理与 HDF5 归档 |
| `dataset/review/` | 问题审核与动作审核 |
| `tools/` | 跨数据、训练和环境边界复用的通用检查工具 |
| `train/bc/` | LoRA 行为克隆数据适配与视觉 SFT |
| `train/rlhf/` | on-policy rollout 契约、统一审核与强化训练 |
| `train/objectives/` | 行为克隆、相对优势与联合目标的纯函数 |
| `game_environment/` | CraftGround 运行时、闭环服务和内存快照 |
| `tests/` | 保留模块的自动化测试 |

## 安装

下面使用 Python 3.13 创建 Linux 训练环境。

```bash
# 安装系统依赖
sudo apt update
sudo apt install -y \
  curl \
  git \
  ffmpeg \
  openjdk-21-jdk \
  python3-pip \
  libgl1 \
  libglib2.0-0 \
  libgl1-mesa-dev \
  libegl1-mesa-dev \
  libglew-dev \
  libglu1-mesa-dev \
  xorg-dev \
  libglfw3-dev \
  xvfb

# CraftGround、Fabric 和首次 Gradle 构建需要 JDK 21
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"
java -version

# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
. "$HOME/.local/bin/env"

# 下载项目并创建虚拟环境
git clone https://github.com/OopsYouDiedE/tao-not-42-base-refactor-world-model-contract.git
cd tao-not-42-base-refactor-world-model-contract
```

创建虚拟环境后，直接按包名安装全部 Python 依赖：

```bash
uv venv --python 3.13
uv pip install unsloth accelerate av craftground datasets gradio h5py huggingface-hub lmdb opencv-python-headless peft pillow pytest ruff transformers trl
```

## 基本用法

下载必要模态：

```bash
python -m dataset.extraction.minestudio.download \
  --dataset 10xx \
  --modality action meta_info image \
  --output-dir runs/datasets
```

默认使用 LMDB 流式训练，不生成中间图片数据集：

```bash
python -m train.bc.gemma_vision_sft \
  --dataset-dir runs/datasets/minestudio-data-10xx-v110 \
  --output-dir runs/trains/gemma-tap
```

HDF5 轨迹题训练使用相同入口，`--dataset-dir` 直接传 `.h5` 或 `.hdf5` 文件。
正式 BC 使用 768 条归档，并按固定种子划分为 691 条训练样本和 77 条验证样本：

```bash
python -m train.bc.gemma_vision_sft \
  --model unsloth/gemma-4-26B-A4B-it \
  --dataset-dir runs/datasets/minestudio-trajectory-sft-768.h5 \
  --output-dir runs/trains/minestudio-gemma4-26b-a4b-bc \
  --lora-rank 32 \
  --micro-batch 4 \
  --gradient-accumulation 2 \
  --epochs 10 \
  --validation-ratio 0.1 \
  --split-seed 3407 \
  --early-stopping-patience 2
```

## 无本地模型的教师流程

以下入口让 Codex 教师从同一 CraftGround 快照独立生成 8 条轨迹，严格执行 TAP v1，
再由独立教师会话统一匿名评分并选择前 4 条。产物进入 BC 数据集；由于教师轨迹没有
行为策略的逐 token 概率，流程会记录并跳过 RLHF 训练。

```bash
python -m tools.audits.codex_teacher_batch8 \
  --runtime /path/to/craftground-runtime \
  --output runs/codex-teacher-batch8 \
  --codex-model MODEL_NAME
```

验证 loss 用于早停与选择最佳 checkpoint，不额外划分测试集。

启动 CraftGround 闭环服务：

```bash
python -m game_environment.closed_loop_server \
  --runtime /path/to/patched/craftground-runtime \
  --output runs/craftground-closed-loop \
  --port 18400
```

## 验证

```bash
python -m pytest
ruff check .
```

CraftGround 内存快照需要安装项目提供的 Kotlin 扩展，具体步骤见
`game_environment/craftground_mod/README.md`。
