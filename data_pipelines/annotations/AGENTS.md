# 动作语言标注（Sonnet action annotation）

用视觉大模型（默认 Claude Sonnet）给 MineStudio 窗口的**真值动作**补一层**语言化理由**，
产出 `Type/Key/Action/Explanation` 结构化标注，作为行为克隆的辅助监督（视觉思维链）。

## 核心原则：动作是硬事实，模型只补理由

MineStudio 数据自带玩家真实键鼠记录（VPT 真值）。**不要**让模型去"猜"动作替代真值——那是
拿弱标注覆盖精确标签、蒸馏方向反了。这里模型只做一件事：**解释这个已发生的动作在当前画面
语境下为何合理**。真值动作原样喂给模型，模型输出理由，二者一致性天然更高。

- **忽略抖动**：相机微偏（mu-law bin 偏离中性 `|Δ|≤1`）与偶发无意义键不算关键动作，
  由 `describe_action` 在喂给模型前就滤掉，模型看到的已是去抖后的动作。
- **理由必须 grounding 到画面**：Explanation 要引用画面里看得到的东西（地形、方块、生物、
  界面），不能脱离画面空想。矛盾标注（画面无敌人却说"躲避攻击"）比不标更糟。
- **教师≥学生**：标注模型应不弱于被训练的策略；默认 `claude-sonnet-5`。

## 采样约定

- **画面 5fps**：MineStudio 原始 20fps，每 `stride=4` 帧取 1 帧画面。
- **动作聚合**：那 4 帧的真值动作聚合成一个 5fps 动作——二值键取 OR（窗口内按下过即算），
  相机取偏离中性最大的分量（保留主导转向，滤掉往返抖动）。见 `aggregate_actions`。

## 如何生成

```bash
export ANTHROPIC_API_KEY=...          # 不硬编码；脚本从环境变量读
python -m data_pipelines.annotations.sonnet_action_annotator \
    --data-directory runs/data/minestudio/10xx \
    --dataset-group 10xx \
    --model claude-sonnet-5 \
    --windows 20 --sampled-steps 8 \
    --output runs/annotations/minestudio_10xx.jsonl
```

输出 JSONL，每行一个窗口：`window_index / task / fps / actions（去抖后真值动作串）/
annotation（模型产出的标注全文）`。

## 案例（真实窗口 10xx #1370，像素统计核验）

下面的案例基于真实导出的帧图(`runs/annotate_sample/step*.png`)与 `actions.json` 真值序列。
帧图可在该路径直接查看。像素级统计:
- **顶部 20% 区域**平均 R60-70/G65-71/B49-54 → 暗绿色（户外白天，树叶或草地）。
- **中心区域**平均 R106-150/G93-124/B63-84 → 暖棕黄色（R>G>B 差值 ~60，木材/泥土类方块；
  若是石头则 R≈G≈B 差值<20）。
- **底部 10%** 为游戏 HUD/热键栏 UI 覆盖层。
- 整段 pitch+ 相机偏移（多帧在向下看），与 `use` 动作共现 → 玩家俯视脚下方块做交互操作。

任务目标：`obtain a diamond pickaxe through the Minecraft technology tree`。
去抖阈值：`|Δbin| ≤ 1`（`CAMERA_JITTER_BIN = 1`）。

**Step 0 —— 真值动作：`use`**（原始 `use cam(+1,-1)`，相机两分量均为 ±1 属抖动，已滤除）
```
Scene: 户外白天；视角偏低；中心暖棕色方块（木材/泥土类）；热键栏可见。
Type: mouse
Key: right_click
Action: press
Explanation: 中心区域为可与之交互的方块（暖色木材/泥土类），use（右键）在 Minecraft 中
             用于放置或交互；相机偏移量 ±1 属抖动，视线基本稳定对准中心方块。
```

**Step 6 —— 真值动作：`back, strafe-right, hotbar-2`**（原始 `B R h2 cam(+1,+0)`，相机 yaw+1 属抖动）
```
Scene: 户外白天；中心棕黄色方块；热键栏高亮格切换。
Type: keyboard
Key: s
Action: hold
Explanation: 后退（s）使玩家离开当前所站位置，常见于放置方块后拉开距离以便瞄准下一位置。
Type: keyboard
Key: d
Action: hold
Explanation: 向右横移（d）结合后退，调整与中心方块的相对位置。
Type: keyboard
Key: 2
Action: press
Explanation: 热键栏切换到第 2 格更换手持物品，对应底部 HUD 可观测的高亮格变化。
```

> **注**：案例由本项目开发者基于像素统计与真值动作联合推断产出，非外部模型的真实标注输出。
> 实际生产标注须由 `sonnet_action_annotator.py` 调视觉大模型完成，使模型真正看到画面后再输出理由。
> 帧图路径：`runs/annotate_sample/`；真值序列：`runs/annotate_sample/actions.json`。
