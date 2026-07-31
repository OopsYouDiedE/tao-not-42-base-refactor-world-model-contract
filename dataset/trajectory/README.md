# MineStudio 轨迹微调数据

本目录负责生成四类轨迹题、定义训练协议、打包双审通过的 HDF5，并把 HDF5 加载成视觉 SFT
对话。候选题、审核记录和训练归档均写入 `runs/`。

## 模块

| 文件 | 职责 |
|---|---|
| `generate_questions.py` | 生成候选题、图片、隔离答案、审核请求并执行自动质量过滤 |
| `sft_protocol.py` | 规范题面、意图、assistant 动作优先输出和训练理由 |
| `pack_hdf5.py` | 校验双审结果并写入自包含 HDF5 |
| `load_hdf5.py` | 从 HDF5 恢复图片和多图对话 |
| `examples/images/` | 不依赖 `runs/` 的协议示例图片 |

人工审核入口位于同目录的 `review_questions.py` 和 `review_actions.py`。

## 题型

| 题型 | 模型输入 | 监督目标 |
|---|---|---|
| `demonstration_optimization` | 图像序列和原始动作 | 经双审的去噪动作轨迹 |
| `image_sequence_to_action` | 完整状态变化图像序列 | 每对相邻图像对应的动作块 |
| `history_to_future_action` | 过去图像序列 | 自行选择时长的未来动作块 |
| `single_frame_intent_to_action` | 当前图像和文字意图 | 推进意图的未来动作块 |

重建题公开每个相邻图像区间的 tick 数。未来规划题不公开参考答案长度，由模型按动作类型选择，
持续移动、挖掘、攻击、拉弓、进食和连续使用最多为 60 tick。

assistant 首先输出可独立解析的 JSON 动作数组，然后另起一行输出 `Reason:`：

```text
["<|action_start|> ; W ; Mouse 4 -2 W ; W <|action_end|>"]
Reason: visible evidence and duration choice
```

每个分号表示一个 50 ms tick。`Mouse dx dy` 表示当前 tick 的相对鼠标移动。GUI 点击使用按下沿
脉冲，普通画面的持续挖掘保留连续 `MouseLeft`。只有需要同 tick 执行时才混写鼠标与按键。

## 生成

```bash
python -m dataset.trajectory.generate_questions \
  --dataset-dir runs/datasets/minestudio-data-10xx-v110 \
  --output-dir runs/datasets/minestudio-trajectory-candidates \
  --samples-per-type 100 \
  --seed 20260730
```

追加其他来源时复用输出目录并传 `--append`。生成器会从已有题号继续，写入后重新验证全部 JSONL、
图片和动作块。

自动过滤覆盖以下明显问题：

| 原因 | 检查目标 |
|---|---|
| `image_too_dark` | 极暗画面 |
| `gui_state_transition_inside_context` | 历史窗口跨越 GUI 与普通游戏状态 |
| `camera_outlier` | 极端鼠标位移 |
| `insufficient_visual_change` | 视觉反推题几乎没有状态变化 |
| `gui_change_without_click` | GUI 变化缺少点击动作 |
| `static_or_weak_action` | 静止或只有无意义微小鼠标运动 |

自动规则用于拒绝确定性坏题，不能替代视觉双审。

## 双审

第一轮人工审核题目设计、图像可作答性、来源区间和原始参考动作：

```bash
python -m dataset.trajectory.review_questions \
  --dataset-dir runs/datasets/minestudio-trajectory-candidates \
  --raw-dataset-dir runs/datasets/minestudio-data-10xx-v110 \
  --host 127.0.0.1 \
  --port 7860
```

第二轮只加载第一轮通过题，审核最终动作答案：

```bash
python -m dataset.trajectory.review_actions \
  --dataset-dir runs/datasets/minestudio-trajectory-candidates \
  --raw-dataset-dir runs/datasets/minestudio-data-10xx-v110 \
  --host 127.0.0.1 \
  --port 7861
```

准入条件：

| 条件 | 要求 |
|---|---|
| AI 审核 | `decision=approve` |
| 人工审核 | `decision=approve` |
| 可选评分 | 提供时每项至少为 3 |
| 最终答案 | AI 与人工 `reviewed_answer_sequence` 一致 |
| 演示优化类型 | `reviewed_optimized_demonstration` |
| 其他题型类型 | `reviewed_optimized_action_sequence` |

审核记录只保存决定和简短依据，不保存模型隐藏推理。

## HDF5

```bash
python -m dataset.trajectory.pack_hdf5 \
  --dataset-dir runs/datasets/minestudio-trajectory-reviewed \
  --ai-reviews runs/datasets/minestudio-trajectory-reviewed/ai_reviews.jsonl \
  --human-reviews runs/datasets/minestudio-trajectory-reviewed/human_reviews.jsonl \
  --output runs/datasets/minestudio-trajectory-train.h5
```

归档为每个样本保存规范题面 JSON、最终答案 JSON 和原始 JPEG 字节。训练时不依赖外部图片目录。

```python
from pathlib import Path

from dataset.trajectory.load_hdf5 import load_hdf5_conversations

conversations = load_hdf5_conversations(
    Path("runs/datasets/minestudio-trajectory-train.h5"),
    maximum_samples=1000,
)
```

## 训练

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

训练入口根据 `.h5` 或 `.hdf5` 后缀自动选择 HDF5 加载器。user message 中图片位于文本之前，
assistant message 以完整动作 JSON 数组开头。
768 条正式归档固定划分为 691 条训练样本和 77 条验证样本；验证集用于早停与最佳 checkpoint
选择，不划分测试集。
