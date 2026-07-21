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

## 案例（真实窗口 10xx #1370，看图核验）

案例基于真实导出帧图（`runs/annotate_sample/step*.png`，拼图 `mosaic.png`）与
`actions.json` 真值序列。场景：户外暗绿色地面上摆着浅色**橡木木板**（同心方形木纹）排成
加号/菱形，玩家俯视地面逐块放置、绕着阵列后退调整视角，底部为热键栏 HUD。任务目标
`obtain a diamond pickaxe through the Minecraft technology tree`。去抖阈值 `|Δbin| ≤ 1`
（`CAMERA_JITTER_BIN = 1`）。完整 8 步标注见 `runs/annotate_sample/annotation_reference.md`。

**Step 0 —— 真值动作：`use`**（原始 `use cam(+1,-1)`，相机两分量均 ±1 属抖动，已滤除）
```
Scene: 俯视地面，3-4 块橡木木板排成 L/角形，中心一块颜色偏亮（刚放下）。
Type: mouse
Key: right_click
Action: press
Explanation: 准星对准地面木板阵列的空位，use（右键）用于放置方块，正在把橡木木板补进图案。
```

**Step 6 —— 真值动作：`back, strafe-right, hotbar-2`**（原始 `B R h2 cam(+1,+0)`，相机 yaw+1 属抖动）
```
Scene: 继续后退，木板阵列偏左，右下有独立方块，热键栏高亮格切换。
Type: keyboard
Key: s
Action: hold
Explanation: 后退（s）拉开与已放方块的距离，绕到下一个放置点。
Type: keyboard
Key: d
Action: hold
Explanation: 右移（d）横向对齐到阵列右侧，配合后退调整站位。
Type: keyboard
Key: 2
Action: press
Explanation: 切换热键栏第 2 格更换手持物品，对应底部 HUD 高亮格变化。
```

> **注**：上述案例为看图 + 真值动作联合产出的**参考标注**（供人理解格式与质量基线），
> 生产标注由 `sonnet_action_annotator.py` 调视觉大模型批量生成。像素统计（顶部暗绿 R60-70/
> G65-71/B49-54=草地、中心暖棕 R>G>B 差值~60=木材类）可作独立交叉核验，见 `../README.md`。
