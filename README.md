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

数据构建环境：

```bash
uv venv --python 3.11
uv pip install -e .
```

按用途安装可选依赖：

```bash
uv pip install -e ".[review]"       # Gradio 审核界面
uv pip install -e ".[train]"        # CUDA、Unsloth 与视觉 SFT
uv pip install -e ".[craftground]"  # 闭环执行与快照验证
uv pip install -e ".[dev]"          # pytest 与 Ruff
```

训练机器通常同时安装数据与训练依赖：

```bash
uv pip install -e ".[review,train,dev]"
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
