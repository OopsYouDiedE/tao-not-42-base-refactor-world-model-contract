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

## 地图模块(EgoMap / MapQuery)

| 部件 | 位置 | 谁 import | 运行时? | 现状 |
|---|---|---|---|---|
| EgoMapNorthLoc / EgoMapClip / EgoMapNaive / EgoMapNorth / MapQuery | `net/fovea_twotower/ego_map.py` | 仅 `train/fovea_twotower/map_probe.py:24`、`map_loc_probe.py:27`、`tests/integration/assembly_a1.py:25`、`map_approach_ablation.py:21` | 否 | 无任何策略读过它。探针结论:自定位 0.33×dead(G-loc1 PASS);clip 3 级嵌套 75% 预算下近场无损、覆盖半径 2×;MapQuery 严口径 0.27 FAIL(方位分量,最近实例歧义) |

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

- `net/dreamer4/`、`net/dreamerv3/`、`net/dreamer/`、`net/ppo_ad/`、`net/bc/`、
  `net/guidance/`、`net/encoders/`、`net/dino_tokenizer.py`、`net/backbone.py` — 均未接入。
- `net/vpt_lib/` — **代码未 import**,但需与"口径被沿用"区分:mu-law 动作编解码这一**设计口径
  被间接沿用**(grpo_pixel.py:205-210 `bins_to_deg` 内联了 mu=8.0 的 mu-law,与
  `train/minecraft/vpt_action.py` 同源),vendored 的 vpt_lib 网络本体则未上线。
- `net/fovea_twotower/` 的感知+快塔子树(`tower.py`、`token_stream.py`、`yolo_unified.py`、
  `yolo_parse.py`、`seg_head.py`、`wood.py`)— 按 grpo_pixel.py:6-9 裁决退役,不在像素运行时路径上
  (ego_map.py 亦在此目录,见上表)。

## 宏技能层

| 部件 | 位置 | 运行时? | 现状 |
|---|---|---|---|
| 挖掘宏(raycast 闩锁,iron_ore ≤5.5 格) | `tests/integration/fullloop_chain.py:170`;另见 `capture_wood_traj.py:125`、`map_approach_ablation.py:60` | 否 | 只活在 integration 脚本里 |
| GUI 合成宏 | — | 否 | 从未建过;仅 `craft_skill.py` 有可脚本化教师,无宏层。grpo_pixel 无任何宏 |
