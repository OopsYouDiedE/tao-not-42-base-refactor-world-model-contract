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

## 运行时路径实际构成(从 grpo_pixel.py 的 import 画出;2026-07-10 v2 接线后)

- 入口:`train/craftground/grpo_pixel.py`(`--tower v1|v2` 双实现,默认 v1)。
- v1 快塔:`net/pixel_tower.py`(grpo_pixel.py import `build_pixel_tower`),依赖
  `blocks/attention.py::MHABlock`。
- v2 快塔(2026-07-10 接线):`train/craftground/tower_v2.py`(grpo_pixel.py import
  V2Config/V2Policy/V2Runtime/DinoFrontend/v2_replay)→ `net/token_tower.py` +
  `net/map_io.py` + `net/fovea_twotower/ego_map.py::EgoMapClip` + `net/backbone.py`
  (冻结 DINOv3 ViT-S/16)+ `net/calibration.py`(符号/FOV/步速自标定)。
- 优势函数:`train/fovea_twotower/grpo_harness.py`,只 import `group_advantage`。
- 外部服务(非本仓权重):慢塔 vLLM(OpenAI 客户端)、Haiku 判官(`claude` CLI 子进程)、
  MiniLM 文本编码(sentence_transformers,v1 goal 向量;v2 语言通道走 UTF-8 字节不经它)、
  CraftGround 环境。

## 2026-07-10 v2 接线完成项(原"建成未接线"表,已全部入链)

| 部件 | 位置 | 接线状态 |
|---|---|---|
| ipm_ground / MapWriter / MapReader / AimPin | `net/map_io.py` | **已接线**:V2Runtime.tick 每 tick 把 DINO patch 按中心 uv 经 ipm_ground 稠密落地写入 EgoMapClip(GPU),MapReader 读出 48 个地图 token 进 KV;aim 经 AimPin 钉世界系(B1)。单测 `test_map_io.py` 5/5 + `test_tower_v2.py`(写图真发生/降级/账本) |
| TokenPolicyTower | `net/token_tower.py` | **已接线**:`grpo_pixel --tower v2` 选择;新增 n_frames 帧堆叠(frame_emb 注帧身份,默认 1 向后兼容)。语言 token=subgoal UTF-8 字节,慢塔刷新换血。单测 `test_token_tower.py` 4/4 + `test_tower_v2.py` 7/7 |
| DINO 前端 | `net/backbone.py`(dinov3 预设)+ `train/craftground/tower_v2.py::DinoFrontend` | **已接线**:DINOv3 ViT-S/16 gated 权重经 HF_TOKEN 实测可加载;96×160 → 60 patch token/帧,S=2 |
| yaw/pitch 符号标定 | `net/calibration.py`(yaw_sign/pitch_sign 派生属性 + fit_angle_map) | **已接线**:部署侧符号取光流增益符号(几何普适);训练侧 env-pose 角度映射由探针采集器实测(runs/probe_aim/pose_calib.json)。测不出置 None ⇒ v2 显式降级(不写图/不钉点/有效位 0) |
| DINO 瞄准可学性探针 | `tests/probe_dino_aim.py` | **已跑,判决 PASS**(104 样本随机 5 折 R²=0.899,hole/slope 0.88 不塌;留 seed 折 0.241 仅 5 场景;数字入档 next_session §2-2) |

v2 的 GRPO 更新按记录 token 回放(采样 π=更新 π 单测锚定);**MapWriter.w_c 与
MapReader.proj 在 GRPO 路径不更新**(记录值当常量),其梯度需 BC 阶段同图重放写读
——v2 的 BC 暖启动是下一个未接线项。v1/v2 checkpoint 分文件(tower.pt / tower_v2.pt),
互不污染;`--init-from` 对现行 bc_vpt checkpoint 的 v1 兼容有单测
(`test_bc_checkpoint_still_loads_into_v1`)。

接线验收(2026-07-10):单测 `tests/unit/` 48/48 通过(新增 test_tower_v2 7 项);
`--tower v2 --smoke` 活环境链路 PASS(判官真排序、slow_fail=0、自标定全实测、
更新执行、tower_v2.pt 落盘,metrics.jsonl 带 tower=v2 字段)。

慢塔设计 2 契约(prev_done/decision/subgoal/aim/done_when + 状态行)**已接线**进
`grpo_pixel.py`(解析/状态行有 `tests/unit/test_slow_contract.py` 4/4;真实 Omni
格式合规率属大模型验证,未跑)。

## 地图模块(EgoMap / MapQuery)

| 部件 | 位置 | 谁 import | 运行时? | 现状 |
|---|---|---|---|---|
| EgoMapClip(+EgoMapNorth 基类) | `net/fovea_twotower/ego_map.py` | `net/map_io.py`(现行地图 IO)→ tower_v2(运行时) | **是(v2)** | 2026-07-10 随 v2 接线,策略经 MapReader 读它;状态张量可驻 GPU(device 参数,向后兼容 CPU) |
| EgoMapNorthLoc / EgoMapNaive / MapQuery / relocalize | 同上 | `train/fovea_twotower/map_probe.py`、`map_loc_probe.py` | 否 | 探针结论:自定位 0.33×dead(G-loc1 PASS);MapQuery 严口径 0.27 FAIL(方位分量,最近实例歧义)。relocalize 周期修正与 MapQuery 慢塔 MAP 行仍未接线 |

## GRPO 长程 harness

| 部件 | 位置 | 谁 import | 运行时? | 现状 |
|---|---|---|---|---|
| `score_rollout`(三层里程碑深度分) | grpo_harness.py:36 | grpo_r1.py:17(旧实现,非运行时) | 否 | 死代码(相对 grpo_pixel) |
| `launch_gate`(启动门) | grpo_harness.py:58 | grpo_r1.py:17(旧实现) | 否 | 死代码 |
| `grpo_update`(优势加权 BC 骨架) | grpo_harness.py:68 | 无 | 否 | 从未被调用 |
| 意图一致性 / 全败组条款 / MILESTONES 链 | grpo_harness.py:22-49 | — | 否 | 三层过程优势设计只在冒烟里干跑过;grpo_pixel 用 Haiku 判官排序代替,只借 `group_advantage` 的 z 归一 |

## 慢塔侧 LoRA adapter(两个,均未接)——**已删除(prune3)**

两个 adapter 训练器(`heartbeat_sft.py`、`reason_delta_sft.py`)及其非运行时消费方
(`fullloop_chain.py`、`m_iron.py`、`grpo_rollout_worker.py`)均无任何运行时消费方,
已于 prune3 物理删除(git 历史可查)。当前慢塔直接用 Omni 出 JSON(grpo_pixel `SlowTower`,
不加载任何 LoRA);QLoRA 工具链结论的复现锚保留在 `train/fovea_twotower/nano9b_qlora_smoke.py`。

## net/ 下运行时未触及的整棵子树

均不被 grpo_pixel / pixel_tower / grpo_harness import:

- ~~`net/dreamer4/`、`net/dreamerv3/`、`net/dreamer/`、`net/ppo_ad/`、`net/bc/`、
  `net/guidance/`~~ — **已删除(2026-07-10 清理,连同其 train/crafter、retired trainers
  与 dreamer 系 tests;git 历史可查)**。
- `net/encoders/`、`net/dino_tokenizer.py`、`net/backbone.py` — 未接入但保留
  (DINO vs YOLOE 裁决探针与 C3-DINO 前端候选,见 design_bitter_lesson §8)。
- ~~`net/vpt_lib/`~~ — **已删除(prune3,全库零 import)**。需与"口径被沿用"区分:mu-law
  动作编解码这一**设计口径被间接沿用**(action_contract.py `bins_to_deg` 内联了 mu=8.0 的
  mu-law,与 `train/minecraft/vpt_action.py` 同源),vendored 的 vpt_lib 网络本体从未上线,
  需要时重拉上游 `openai/Video-Pre-Training`。
- ~~`net/fovea_twotower/` 的感知+快塔子树(`tower.py`、`token_stream.py`、`yolo_unified.py`、
  `yolo_parse.py`、`seg_head.py`、`wood.py`)~~ — **已删除(2026-07-10 用户拍板 DINO,
  YOLOE 整线废弃;连同 train/fovea_twotower 44 个退役训练器、9 个 integration 脚本、
  yolo_backbone_encoder、train_ppo_ad 等,git 历史可查)**。目录仅存 ego_map.py(现行)。
  保留的 train/fovea_twotower:grpo_harness(运行时)、map_probe/map_loc_probe(地图现行)、
  judge_exam 系 + judge_train(判官对照纪律 + E3 本地 RM 锚)、nano9b_qlora_smoke
  (QLoRA 工具链结论的复现锚)。

## 宏技能层——**脚本已删(prune3)**

挖掘宏(raycast 闩锁)与可脚本化 GUI 合成教师只活在 `tests/integration/` 脚本里
(`craft_skill.py`、`fullloop_chain.py` 等),整个 `tests/integration/` 目录已于 prune3
物理删除(已死:内部还 import 更早删掉的 `collect_s8.py`/`skill_ceiling.py`,不可运行)。
grpo_pixel 运行时无任何宏;运行时零脚本纪律沿用。技能天花板结论入档
`knowledge/conclusion_fasttower_skill_ceiling.md`。
