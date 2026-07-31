# LumineCraft

本项目使用 MineStudio Minecraft 轨迹训练视觉动作模型，并通过 CraftGround 执行 Lumine
命名动作协议。仓库只保存源码、测试和稳定契约。数据集、图片、审核结果、日志与模型权重统一写入
`runs/`，该目录不进入版本控制。

## 当前管线

| 阶段 | 输入 | 正式入口 | 输出 |
|---|---|---|---|
| 下载 | Hugging Face MineStudio 数据 | `dataset.minestudio.download` | LMDB 模态分片 |
| 预训练数据 | LMDB 图像与动作 | `dataset.pretraining` | Lumine JSONL 与图片 |
| 流式训练数据 | LMDB 图像与动作 | `train.lumine_streaming_dataset` | 内存中的视觉对话样本 |
| 轨迹题生成 | LMDB 轨迹 | `dataset.trajectory.generate_questions` | 候选题、隔离答案和审核请求 |
| 人工双审 | 候选题与原始轨迹 | `dataset.trajectory.review_questions`、`review_actions` | 题目审核与规范动作答案 |
| 训练归档 | 双审通过题目 | `dataset.trajectory.pack_hdf5` | 自包含 HDF5 |
| 视觉 SFT | LMDB、落盘 JSONL 或 HDF5 | `train.gemma_vision_sft`、`train.qwen_vision_sft` | LoRA adapter |
| 闭环执行 | Lumine 动作文本 | `game_environment.closed_loop_server` | 逐 tick RGB 与轨迹 JSON |

`dataset/trajectory/README.md` 说明轨迹题协议、双审准入条件和 HDF5 训练格式。
`docs/game_environment_reset.md` 说明当前 CraftGround 内存快照边界。
`docs/craftground_closed_loop.md` 说明闭环 HTTP 契约。

## 代码结构

| 路径 | 职责 |
|---|---|
| `lumine/` | Lumine 动作协议与执行契约 |
| `dataset/minestudio/` | MineStudio 下载和 LMDB 读取 |
| `dataset/trajectory/` | 轨迹题生成、双审、协议、HDF5 打包与加载 |
| `dataset/split.py` | 玩家或 episode 级训练集、验证集划分 |
| `dataset/pretraining.py` | 可选的落盘预训练数据构建 |
| `tools/` | 跨数据、训练和环境边界复用的通用检查工具 |
| `train/` | 视觉 SFT、组内相对优势、行为克隆与联合训练目标 |
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
python -m dataset.minestudio.download \
  --dataset 10xx \
  --modality action meta_info image \
  --output-dir runs/datasets
```

默认使用 LMDB 流式训练，不生成中间图片数据集：

```bash
python -m train.gemma_vision_sft \
  --dataset-dir runs/datasets/minestudio-data-10xx-v110 \
  --output-dir runs/trains/gemma-lumine
```

HDF5 轨迹题训练使用相同入口，`--dataset-dir` 直接传 `.h5` 或 `.hdf5` 文件。
正式 BC 使用 768 条归档，并按固定种子划分为 691 条训练样本和 77 条验证样本：

```bash
python -m train.gemma_vision_sft \
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
