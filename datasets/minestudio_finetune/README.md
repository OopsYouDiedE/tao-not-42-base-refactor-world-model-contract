# MineStudio 轨迹微调数据

本目录只负责从 MineStudio 轨迹生成三类候选题、提供 AI 与人工审核标准、把最终准入题打包为
HDF5，以及把 HDF5 解析成视觉 LoRA 训练所需的多图对话。候选 JSONL 写入命令指定的外部
输出目录，不保存在本源码目录。

## 目录

| 路径 | 作用 |
|---|---|
| `generate_questions.py` | 生成候选题、图片、隔离答案和双审请求；包含自动质量过滤与审核提示词 |
| `pack_hdf5.py` | 只把双审通过的题目、答案和 JPEG 打包为一个 HDF5 |
| `load_hdf5.py` | 从 HDF5 恢复图片并生成 Unsloth 多图 `messages` |
| `examples/` | 保存本流程真实生成的案例图片 |
| `AGENTS.md` | 本目录后续维护约束 |

## 三类题目

| 题型 | 输入 | 监督目标 |
|---|---|---|
| `demonstration_optimization` | 图像序列与原始动作 | 经独立审核的去噪动作轨迹 |
| `image_sequence_to_action` | 完整状态变化图像序列 | 一种能够解释变化的动作轨迹 |
| `history_to_future_action` | 过去图像序列 | 未来 200 ms 的一种合理动作轨迹 |

动作块允许变长，每个 `;` 分隔一个 50 ms tick。`Mouse dx dy` 在普通画面表示相机相对
移动，在 GUI 表示光标相对移动。GUI 的鼠标键使用按下沿脉冲；普通画面的持续挖掘继续使用
连续 `MouseLeft`。只有需要同 tick 执行时才把鼠标移动与按键混写。

## 生成与自动过滤

```bash
python -m datasets.minestudio_finetune.generate_questions \
  --dataset-dir runs/datasets/minestudio-data-10xx-v110 \
  --output-dir runs/datasets/minestudio-trajectory-candidates \
  --samples-per-type 10000 \
  --seed 20260730
```

生成器在保存候选题之前过滤以下明显问题：

| 原因 | 判定目标 |
|---|---|
| `image_too_dark` | 极暗画面 |
| `gui_state_transition_inside_context` | 历史窗口跨越 GUI 与普通游戏状态 |
| `camera_outlier` | 极端鼠标位移 |
| `insufficient_visual_change` | 视觉反推题几乎没有状态变化 |
| `gui_change_without_click` | GUI 明显变化，但目标动作没有鼠标点击 |

输出目录包含 `questions.jsonl`、隔离的 `answer_key.jsonl`、`ai_review_requests.jsonl`、
`human_review_templates.jsonl`、图片和生成报告。自动规则只减少明显坏题，不能替代视觉双审。

## AI 与人工审核

AI 和人工提示词分别是 `generate_questions.py` 中的 `AI_REVIEW_PROMPT` 与
`HUMAN_REVIEW_PROMPT`。审核必须查看全部原图，并检查来源完整性、视觉可作答性、动作依据、
演示质量、动作协议和隐私。任一维度低于 3 分即不准入。

演示优化题还有一条硬规则：监督答案必须经过独立清理与双审，且
`reference_kind` 必须写为 `reviewed_optimized_demonstration`。原始人类轨迹只能作为待优化输入。

最终审核流程应把模型和人工填写结果分别保存为：

| 文件 | 要求 |
|---|---|
| `ai_reviews.jsonl` | AI 决定、各维度分数、理由和建议修订 |
| `human_reviews.jsonl` | 人工决定、各维度分数；优化题还要填写 `reviewed_answer_sequence` |

打包器会把双审通过的候选题在 HDF5 内标记为 `approved`。审核文件不得包含模型隐藏推理。

## 打包 HDF5

```bash
python -m datasets.minestudio_finetune.pack_hdf5 \
  --dataset-dir runs/datasets/minestudio-trajectory-reviewed \
  --ai-reviews runs/datasets/minestudio-trajectory-reviewed/ai_reviews.jsonl \
  --human-reviews runs/datasets/minestudio-trajectory-reviewed/human_reviews.jsonl \
  --output runs/datasets/minestudio-trajectory-train.h5
```

打包器逐题检查 AI 与人工均批准、每项分数不低于 3、答案存在、图片存在和优化答案类型。HDF5 内每个样本保存
题面 JSON、答案 JSON 和原始 JPEG 字节，因此训练时不依赖旁边的图片目录。

## 加载与训练

```python
from pathlib import Path
from datasets.minestudio_finetune.load_hdf5 import load_hdf5_conversations

conversations = load_hdf5_conversations(
    Path("runs/datasets/minestudio-trajectory-train.h5"),
    maximum_samples=1000,
)
```

现有 Gemma/Qwen 视觉 LoRA 训练入口已经识别 `.h5` 和 `.hdf5`：

```bash
python -m train.gemma_vision_sft \
  --model gemma-4-26B-A4B-it \
  --dataset-dir runs/datasets/minestudio-trajectory-train.h5 \
  --output-dir runs/trains/minestudio-trajectory-lora \
  --lora-rank 32 \
  --epochs 1
```

HDF5 路径会自动关闭 LMDB 流式加载并使用 `load_hdf5_conversations()`。每个 user message 中
图片位于文本之前，assistant message 只包含 JSON 动作数组。

## 案例一：GUI 间断点击

| 帧 554 | 帧 558 | 帧 562 | 帧 566 |
|---|---|---|---|
| ![](examples/images/demonstration_optimization_000002_00.jpg) | ![](examples/images/demonstration_optimization_000002_01.jpg) | ![](examples/images/demonstration_optimization_000002_02.jpg) | ![](examples/images/demonstration_optimization_000002_03.jpg) |

```text
<|action_start|> ; Mouse -2 6 MouseLeft ; Mouse -2 6 ; Mouse -5 7 ; Mouse -20 12 <|action_end|>
<|action_start|> ; Mouse -12 6 ; Mouse 3 -7 ; Mouse 16 -18 ; Mouse 8 -5 <|action_end|>
<|action_start|> ; Mouse 2 -1 MouseLeft ; Mouse 1 -2 ;  ; Mouse -4 7 <|action_end|>
```

第一段和第三段各有一次点击脉冲。中间 tick 只移动光标。本案例展示协议，原始动作仍需生成
真正的优化答案后才能训练。

## 案例二：图像序列反推挖掘

| 帧 4606 | 帧 4607 | 帧 4608 | 帧 4609 | 帧 4610 |
|---|---|---|---|---|
| ![](examples/images/image_sequence_to_action_000001_00.jpg) | ![](examples/images/image_sequence_to_action_000001_01.jpg) | ![](examples/images/image_sequence_to_action_000001_02.jpg) | ![](examples/images/image_sequence_to_action_000001_03.jpg) | ![](examples/images/image_sequence_to_action_000001_04.jpg) |

该序列包含镐击、视角变化和方块变化。普通游戏中的连续 `MouseLeft` 表示持续按住挖掘。

## 案例三：历史图像预测继续挖掘

| 帧 2334 | 帧 2338 | 帧 2342 | 帧 2346 |
|---|---|---|---|
| ![](examples/images/history_to_future_action_000001_00.jpg) | ![](examples/images/history_to_future_action_000001_01.jpg) | ![](examples/images/history_to_future_action_000001_02.jpg) | ![](examples/images/history_to_future_action_000001_03.jpg) |

历史序列形成持续挥动工具和目标裂纹趋势，未来继续按住 `MouseLeft` 是有视觉依据的合理答案。
