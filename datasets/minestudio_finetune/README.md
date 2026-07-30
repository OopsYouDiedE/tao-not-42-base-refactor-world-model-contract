# MineStudio 轨迹微调数据

本目录只负责从 MineStudio 轨迹生成四类候选题、提供 AI 与人工审核标准、把最终准入题打包为
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

## 四类题目

| 题型 | 输入 | 监督目标 |
|---|---|---|
| `demonstration_optimization` | 图像序列与原始动作 | 经独立审核的去噪动作轨迹 |
| `image_sequence_to_action` | 完整状态变化图像序列 | 一种能够解释变化的动作轨迹 |
| `history_to_future_action` | 过去图像序列 | 未来 200 ms 的一种合理动作轨迹 |
| `single_frame_intent_to_action` | 当前单帧与文字意图 | 推进该意图的未来 200 ms 动作轨迹 |

人工题目审核界面展示每道题的完整参考动作序列。`history_to_future_action` 和
`single_frame_intent_to_action` 各自只额外展示一张未来关键帧，并将它接在全部输入图像后组成
完整的审核展示序列；未来关键帧仅供审核，不进入模型输入。未来最终帧只允许在这两类延拓题中调整。
`image_sequence_to_action` 的动作区间绑定首尾输入图像，`demonstration_optimization` 使用固定长度的
完整演示区间，二者均不提供独立的最终帧选项。
人工调整帧区间后，审核工具从原始 LMDB 重新读取该区间的全部动作，并同步更新参考动作起止帧、
动作块及其逐块帧号。历史序列延拓题的展示顺序固定为全部历史输入图、唯一未来关键帧。
参考动作按相邻图像节点逐段编码，不使用固定动作长度。例如相邻图像帧号为 10 和 15，该段参考
动作必须恰好包含 5 个 tick；下一对图像节点形成下一段动作。生成校验会逐段比较动作 tick 数与
对应图像帧差，任何一段不一致都会拒绝数据集。

`image_sequence_to_action` 的模型输出为每对相邻输入图像各一个动作块。五张输入图因此输出四个
按时间排列的动作块，不得把题面写成只返回一个动作块。

动作块允许变长，每个 `;` 分隔一个 50 ms tick。`Mouse dx dy` 在普通画面表示相机相对
移动，在 GUI 表示光标相对移动。GUI 的鼠标键使用按下沿脉冲；普通画面的持续挖掘继续使用
连续 `MouseLeft`。只有需要同 tick 执行时才把鼠标移动与按键混写。

## 生成与自动过滤

```bash
python -m datasets.minestudio_finetune.generate_questions \
  --dataset-dir runs/datasets/minestudio-data-10xx-v110 \
  --output-dir runs/datasets/minestudio-trajectory-candidates \
  --samples-per-type 100 \
  --seed 20260730
```

四类任务各生成 100 条时总数为 400。追加另一个来源时使用相同输出目录和 `--append`；生成器
会从已有题号继续编号，并在写入后重新读取全部 JSONL、图片和动作块完成一致性校验。

生成器在保存候选题之前过滤以下明显问题：

| 原因 | 判定目标 |
|---|---|
| `image_too_dark` | 极暗画面 |
| `gui_state_transition_inside_context` | 历史窗口跨越 GUI 与普通游戏状态 |
| `camera_outlier` | 极端鼠标位移 |
| `insufficient_visual_change` | 视觉反推题几乎没有状态变化 |
| `gui_change_without_click` | GUI 明显变化，但目标动作没有鼠标点击 |
| `static_or_weak_action` | 完全静止，或只有不足以体现意图的微小鼠标运动 |

输出目录包含 `questions.jsonl`、隔离的 `answer_key.jsonl`、`ai_review_requests.jsonl`、
`human_review_templates.jsonl`、图片和生成报告。自动规则只减少明显坏题，不能替代视觉双审。

## 10xx、7xx 与 1200 条计划

第一批已经从 10xx 生成四类各 100 条，共 400 条。追加 7xx 时只下载动作、元信息和一个
图像编码分片，不下载 segmentation：

```bash
python -m datasets.minestudio_data.download \
  --dataset 7xx \
  --modality action meta_info image \
  --maximum-image-parts 1 \
  --output-dir runs/datasets

python -m datasets.minestudio_finetune.generate_questions \
  --dataset-dir runs/datasets/minestudio-data-7xx-v110 \
  --output-dir runs/datasets/minestudio-trajectory-1200 \
  --samples-per-type 100 \
  --seed 20260731 \
  --append
```

部分 7xx 图像分片与完整动作/元信息取 episode 交集。若过滤后不足 400 条，生成器会把实际
数量和各题型缺口写入 manifest，而不会用静止或弱意图动作补数。最终 1200 条由后续来源继续追加；
每次追加都会重新验证当前目录内的全部样本。

## AI 与人工审核

AI 和人工提示词分别是 `generate_questions.py` 中的 `AI_REVIEW_PROMPT` 与
`HUMAN_REVIEW_PROMPT`。审核必须查看全部原图，并检查来源完整性、视觉可作答性、动作依据、
演示质量、动作协议和隐私。AI 与人工都只记录批准或拒绝，并给出一条简短理由。分项评分可以
作为中间分析记录，但不是必填字段；若提供评分，任一维度低于 3 分仍不准入。

演示优化题还有一条硬规则：监督答案必须经过独立清理与双审，且
`reference_kind` 必须写为 `reviewed_optimized_demonstration`。原始人类轨迹只能作为待优化输入。

最终审核流程应把模型和人工填写结果分别保存为：

| 文件 | 要求 |
|---|---|
| `ai_reviews.jsonl` | AI 的批准或拒绝决定，以及简短理由 |
| `human_reviews.jsonl` | 人工的批准或拒绝决定，以及简短理由 |

打包器会把双审通过的候选题在 HDF5 内标记为 `approved`。审核文件不得包含模型隐藏推理。
四类题型都必须在打包前补充一致的 AI 与人工 `reviewed_answer_sequence`，保证训练标签使用第二轮规范答案。

题目审核全部完成后，第二轮使用 `tools.trajectory_action_review` 只加载第一轮通过题。该轮审核的是
题目设计和最终回答的准确性。界面同时展示完整图像轨迹、录制真值动作、AI 压缩回答和优化依据。
最终回答只包含题型要求的动作序列及其简短理由。普通游戏中的挖掘、蓄力射箭、移动、进食等持续
按键，其 tick 数承载实际动作时长，不允许缩短；同一持续区间内的非特殊鼠标偏移可以累计合并，
但对应按键 tick 必须保留。GUI 保留点击顺序，把点击之间的光标轨迹合并到操作节点。
人工可以修改动作 JSON 后批准，也可以填写理由否决。第二轮预标注写入
`second_round_preannotations.jsonl`，正式结果写入 `action_reviews.jsonl`。演示优化题通过后写入
`reference_kind=reviewed_optimized_demonstration`，其他题写入
`reference_kind=reviewed_optimized_action_sequence`。

```bash
python -m tools.trajectory_action_review \
  --dataset-dir runs/datasets/minestudio-trajectory-1200 \
  --raw-dataset-dir runs/datasets/minestudio-data-7xx-v110 \
  --host 127.0.0.1 \
  --port 7860 \
  --share
```

## 打包 HDF5

```bash
python -m datasets.minestudio_finetune.pack_hdf5 \
  --dataset-dir runs/datasets/minestudio-trajectory-reviewed \
  --ai-reviews runs/datasets/minestudio-trajectory-reviewed/ai_reviews.jsonl \
  --human-reviews runs/datasets/minestudio-trajectory-reviewed/human_reviews.jsonl \
  --output runs/datasets/minestudio-trajectory-train.h5
```

打包器逐题检查 AI 与人工均批准、双方最终动作一致、可选评分不低于 3、答案存在、图片存在和优化答案类型。HDF5 内每个样本保存
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
  --model unsloth/gemma-4-26B-A4B-it \
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

## 7xx-800 实际训练样本抽样

正式训练包是 `runs/datasets/minestudio-trajectory-7xx-800-train.h5`，共 531 条。以下使用固定随机
种子 `42` 从四类入库题目中各抽一条。内容按 `load_hdf5.py` 的真实训练路径展示：模型先收到
按时间排列的 JPEG，随后收到完整文字 Prompt；assistant 监督目标只包含动作 JSON 数组。

### 演示动作优化：`demonstration_optimization_000188`

| 帧 123 | 帧 127 | 帧 131 | 帧 135 |
|---|---|---|---|
| ![](../../runs/datasets/minestudio-trajectory-7xx-800/images/demonstration_optimization_000188_00.jpg) | ![](../../runs/datasets/minestudio-trajectory-7xx-800/images/demonstration_optimization_000188_01.jpg) | ![](../../runs/datasets/minestudio-trajectory-7xx-800/images/demonstration_optimization_000188_02.jpg) | ![](../../runs/datasets/minestudio-trajectory-7xx-800/images/demonstration_optimization_000188_03.jpg) |

```text
The images and raw action blocks form one chronological Minecraft demonstration. Rewrite it as a cleaner action sequence while preserving visible intent and causal order. Return one block per adjacent image pair and exactly match the supplied tick count for every block. One semicolon is one 50 ms tick. Do not shorten duration-sensitive held actions such as mining, attacking, moving, drawing a bow, eating, or continuous use. Remove only visually unsupported camera jitter; preserve GUI click order. Return only the JSON array of action blocks.
Required action-block tick counts: [4, 4, 4]
Action format example for a 3-tick block: "<|action_start|> ; W ; Mouse 4 -2 W ; W <|action_end|>". Each JSON array item must be one string action block; do not return nested tick arrays.
Raw action sequence:
["<|action_start|> ; Mouse -12 6 MouseLeft ; Mouse 10 24 MouseLeft ; Mouse 12 18 MouseLeft ; MouseLeft <|action_end|>", "<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; Mouse 3 -8 MouseLeft <|action_end|>", "<|action_start|> ; Mouse 3 -14 MouseLeft ; Mouse -6 -11 MouseLeft ; Mouse -5 -7 MouseLeft ; Mouse -3 0 MouseLeft <|action_end|>"]
```

参考输出：

```json
["<|action_start|> ; Mouse -2 30 MouseLeft ; MouseLeft ; MouseLeft ; Mouse 12 18 MouseLeft <|action_end|>", "<|action_start|> ; MouseLeft ; MouseLeft ; MouseLeft ; Mouse 3 -8 MouseLeft <|action_end|>", "<|action_start|> ; Mouse -3 -25 MouseLeft ; MouseLeft ; MouseLeft ; Mouse -8 -7 MouseLeft <|action_end|>"]
```

### 图像序列转动作：`image_sequence_to_action_000036`

| 帧 29515 | 帧 29519 | 帧 29523 | 帧 29527 | 帧 29531 |
|---|---|---|---|---|
| ![](../../runs/datasets/minestudio-trajectory-7xx-800/images/image_sequence_to_action_000036_00.jpg) | ![](../../runs/datasets/minestudio-trajectory-7xx-800/images/image_sequence_to_action_000036_01.jpg) | ![](../../runs/datasets/minestudio-trajectory-7xx-800/images/image_sequence_to_action_000036_02.jpg) | ![](../../runs/datasets/minestudio-trajectory-7xx-800/images/image_sequence_to_action_000036_03.jpg) | ![](../../runs/datasets/minestudio-trajectory-7xx-800/images/image_sequence_to_action_000036_04.jpg) |

```text
The images are consecutive Minecraft observations in chronological order. Infer one reasonable action sequence that produced every adjacent transition. Return only a JSON array containing one valid action block for each adjacent image pair, with each block exactly matching its supplied tick count. One semicolon is one 50 ms tick. Keep movement, mining, attacking, drawing, eating, and continuous use held for the required duration. Use visible camera displacement to infer meaningful mouse direction, omit unsupported 1-2 pixel jitter, and preserve GUI click order.
Required action-block tick counts: [4, 4, 4, 4]
Action format example for a 3-tick block: "<|action_start|> ; W ; Mouse 4 -2 W ; W <|action_end|>". Each JSON array item must be one string action block; do not return nested tick arrays.
```

参考输出：

```json
["<|action_start|> ; Mouse 140 29 W A space ; W A space ; W A ; Mouse 162 31 W A <|action_end|>", "<|action_start|> ; Mouse 53 9 A ; A ; A ; Mouse 44 9 A <|action_end|>", "<|action_start|> ; Mouse 49 1 A ;  ;  ; Mouse 19 -4 <|action_end|>", "<|action_start|> ; Mouse 1 -1 ; MouseLeft ; MouseLeft ; Mouse -29 -20 <|action_end|>"]
```

### 历史帧预测未来动作：`history_to_future_action_000010`

| 帧 667 | 帧 675 | 帧 679 | 当前帧 683 |
|---|---|---|---|
| ![](../../runs/datasets/minestudio-trajectory-7xx-800/images/history_to_future_action_000010_00.jpg) | ![](../../runs/datasets/minestudio-trajectory-7xx-800/images/history_to_future_action_000010_01.jpg) | ![](../../runs/datasets/minestudio-trajectory-7xx-800/images/history_to_future_action_000010_02.jpg) | ![](../../runs/datasets/minestudio-trajectory-7xx-800/images/history_to_future_action_000010_03.jpg) |

```text
The images are past Minecraft observations in chronological order. Infer one reasonable action sequence for the supplied future horizon. Return one valid action block with exactly the supplied number of 50 ms ticks. Continue visually established held actions for a plausible duration, omit unsupported 1-2 pixel camera jitter, and do not invent GUI clicks or auxiliary keys without visual evidence. Return only a JSON array.
Required action-block tick counts: [20]
Action format example for a 3-tick block: "<|action_start|> ; W ; Mouse 4 -2 W ; W <|action_end|>". Each JSON array item must be one string action block; do not return nested tick arrays.
```

参考输出：

```json
["<|action_start|> ; Mouse 3 2 W space ; W space ; W space ; W space ; W space ; W space ; W space ctrl ; W space ctrl ; W space ctrl ; W space ctrl ; W space ; W space ; W space ; W space ; W space ; W space ; W space ; W space ; W space ctrl ; W space ctrl <|action_end|>"]
```

### 单帧意图转动作：`single_frame_intent_to_action_000135`

| 当前帧 454 |
|---|
| ![](../../runs/datasets/minestudio-trajectory-7xx-800/images/single_frame_intent_to_action_000135_00.jpg) |

```text
The image is the current Minecraft observation and the intent is supplied as text. Infer one reasonable action sequence for the supplied future horizon that advances this intent. Return one valid action block with exactly the supplied number of 50 ms ticks. Preserve the required duration of mining, movement, bow drawing, eating, or continuous use; omit unsupported 1-2 pixel camera jitter and preserve GUI click order. Return only a JSON array.
Required action-block tick counts: [40]
Action format example for a 3-tick block: "<|action_start|> ; W ; Mouse 4 -2 W ; W <|action_end|>". Each JSON array item must be one string action block; do not return nested tick arrays.
Intent: 沿当前可见路线疾跑跳跃（39 tick，向右平视修正视角）
```

参考输出：

```json
["<|action_start|> ; Mouse -31 35 W ; W ; W ; W ; W ; W ; W ; W ; W ; W ; W ; W ; W ; W ; W ; W ; W ; W ; W space ; W space ; W space ; W ; W ; W ; W ; W ; W ; W ; W ; W ; W ; W ; W ctrl ; W ctrl ; W ctrl ; W ; W ; W ; W ; Mouse 168 -18 W <|action_end|>"]
```
