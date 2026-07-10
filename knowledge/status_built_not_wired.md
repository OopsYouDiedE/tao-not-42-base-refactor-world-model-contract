---
name: status-built-not-wired
description: 已造好但未接入运行时的部件清单。判据=是否出现在当前唯一在跑的路径 train/craftground/grpo_pixel.py 的 import/调用链上。每条给 file:line,核实过。
metadata:
  type: knowledge
---

# 造好了但没接进系统

接入判据:一个部件算"已接入",当且仅当它出现在当前唯一在跑的运行时路径——
`train/craftground/grpo_pixel.py` 的 import 链与调用链上。下表每条经 Grep 核实,给 `file:line`。
不在这条链上的一律记为"未接入",不管它历史上花了多少功夫。

## 运行时路径实际构成(从 grpo_pixel.py 的 import 画出)

- 入口:`train/craftground/grpo_pixel.py`。
- 快塔:`net/pixel_tower.py`(grpo_pixel.py:56 import `build_pixel_tower`),它只依赖
  `blocks/attention.py::MHABlock`(pixel_tower.py:29)。
- 优势函数:`train/fovea_twotower/grpo_harness.py`,但**只 import 了 `group_advantage` 一个函数**
  (grpo_pixel.py:57)。
- 外部服务(非本仓权重):Omni 慢塔(`tests/serve_omni_nvfp4.sh` 起的 vLLM,OpenAI 客户端调用
  grpo_pixel.py:51,116)、Haiku 判官(`claude` CLI 子进程 grpo_pixel.py:187)、
  MiniLM 文本编码(sentence_transformers,grpo_pixel.py:328)、CraftGround 环境(grpo_pixel.py:322)。

**即运行时只有四个本仓文件:grpo_pixel.py + pixel_tower.py + blocks/attention.py +
grpo_harness.py(仅 group_advantage)。** 以下全部不在其中。

## 2026-07-10 新建(按重设计定稿造,**建成未接线**,接线条件注明)

| 部件 | 位置 | 单测 | 接线条件 |
|---|---|---|---|
| ipm_ground / MapWriter / MapReader / AimPin | `net/map_io.py` | `tests/unit/test_map_io.py` 5/5(IPM 精确几何/写读闭环/W_c 梯度/北锚定运动账本/钉点生命周期) | 随 TokenPolicyTower 接线;yaw/pitch 符号与 CraftGround 的常量标定属训练侧,首次接线时用 env pose 标定 |
| TokenPolicyTower(goal-as-query cross-attn + UTF-8 字节语言 token) | `net/token_tower.py` | `tests/unit/test_token_tower.py` 4/4(形状/各组梯度/反义 token 可分/空组容错) | 视觉前端已拍板 DINO(2026-07-10 用户裁决,YOLOE 整线已删),待接线;A1 语言通道 grounding 需 hindsight relabel BC 数据 |
| DINO 瞄准可学性探针(单臂) | `tests/probe_dino_aim.py`(原 DINO vs YOLOE 双臂,YOLOE 臂随裁决摘除;ridge 对偶式已合成数据验证) | — | 需活环境采 (帧, 准星角偏移, 地形) 清单 `runs/probe_aim/manifest.jsonl`;标签 raycast 只进训练侧 |

慢塔设计 2 契约(prev_done/decision/subgoal/aim/done_when + 状态行)**已接线**进
`grpo_pixel.py`(解析/状态行有 `tests/unit/test_slow_contract.py` 4/4;真实 Omni
格式合规率属大模型验证,未跑)。

## 地图模块(EgoMap / MapQuery)

| 部件 | 位置 | 谁 import | 运行时? | 现状 |
|---|---|---|---|---|
| EgoMapNorthLoc / EgoMapClip / EgoMapNaive / EgoMapNorth / MapQuery | `net/fovea_twotower/ego_map.py` | `net/map_io.py:22`(现行地图 IO)、`train/fovea_twotower/map_probe.py`、`map_loc_probe.py`、`tests/unit/test_map_io.py` | 否 | 无任何策略读过它。探针结论:自定位 0.33×dead(G-loc1 PASS);clip 3 级嵌套 75% 预算下近场无损、覆盖半径 2×;MapQuery 严口径 0.27 FAIL(方位分量,最近实例歧义) |

## GRPO 长程 harness

| 部件 | 位置 | 谁 import | 运行时? | 现状 |
|---|---|---|---|---|
| `score_rollout`(三层里程碑深度分) | grpo_harness.py:36 | grpo_r1.py:17(旧实现,非运行时) | 否 | 死代码(相对 grpo_pixel) |
| `launch_gate`(启动门) | grpo_harness.py:58 | grpo_r1.py:17(旧实现) | 否 | 死代码 |
| `grpo_update`(优势加权 BC 骨架) | grpo_harness.py:68 | 无 | 否 | 从未被调用 |
| 意图一致性 / 全败组条款 / MILESTONES 链 | grpo_harness.py:22-49 | — | 否 | 三层过程优势设计只在冒烟里干跑过;grpo_pixel 用 Haiku 判官排序代替,只借 `group_advantage` 的 z 归一 |

## 慢塔侧 LoRA adapter(两个,均未接)

| 部件 | 位置 | 谁 import/消费 | 运行时? | 现状 |
|---|---|---|---|---|
| 心跳微决策 adapter(1.5B) | `train/fovea_twotower/heartbeat_sft.py`(训练器) | 无人加载 | 否 | 训出的 adapter 无任何运行时消费方 |
| 差额规划 adapter `reason_delta_lora_v4` | 由 `reason_delta_sft.py:218` 训 | 仅 `fullloop_chain.py:64,247`、`m_iron.py:101`、`grpo_rollout_worker.py:66`(全非运行时) | 否 | 当前慢塔直接用 Omni 出 JSON(grpo_pixel `SlowTower`,不加载任何 LoRA) |

## net/ 下运行时未触及的整棵子树

均不被 grpo_pixel / pixel_tower / grpo_harness import:

- ~~`net/dreamer4/`、`net/dreamerv3/`、`net/dreamer/`、`net/ppo_ad/`、`net/bc/`、
  `net/guidance/`~~ — **已删除(2026-07-10 清理,连同其 train/crafter、retired trainers
  与 dreamer 系 tests;git 历史可查)**。
- `net/encoders/`、`net/dino_tokenizer.py`、`net/backbone.py` — 未接入但保留
  (DINO vs YOLOE 裁决探针与 C3-DINO 前端候选,见 design_bitter_lesson §8)。
- `net/vpt_lib/` — **代码未 import**,但需与"口径被沿用"区分:mu-law 动作编解码这一**设计口径
  被间接沿用**(grpo_pixel.py `bins_to_deg` 内联了 mu=8.0 的 mu-law,与
  `train/minecraft/vpt_action.py` 同源),vendored 的 vpt_lib 网络本体则未上线
  (E1 BC 暖启动的数据管线,保留)。
- ~~`net/fovea_twotower/` 的感知+快塔子树(`tower.py`、`token_stream.py`、`yolo_unified.py`、
  `yolo_parse.py`、`seg_head.py`、`wood.py`)~~ — **已删除(2026-07-10 用户拍板 DINO,
  YOLOE 整线废弃;连同 train/fovea_twotower 44 个退役训练器、9 个 integration 脚本、
  yolo_backbone_encoder、train_ppo_ad 等,git 历史可查)**。目录仅存 ego_map.py(现行)。
  保留的 train/fovea_twotower:grpo_harness(运行时)、map_probe/map_loc_probe(地图现行)、
  judge_exam 系 + judge_train(判官对照纪律 + E3 本地 RM 锚)、nano9b_qlora_smoke
  (QLoRA 工具链结论的复现锚)。

## 宏技能层

| 部件 | 位置 | 运行时? | 现状 |
|---|---|---|---|
| 挖掘宏(raycast 闩锁,iron_ore ≤5.5 格) | `tests/integration/fullloop_chain.py:170`;另见 `capture_wood_traj.py:125`、`map_approach_ablation.py:60` | 否 | 只活在 integration 脚本里 |
| GUI 合成宏 | — | 否 | 从未建过;仅 `craft_skill.py` 有可脚本化教师,无宏层。grpo_pixel 无任何宏 |
