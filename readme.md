# TAO-Not-42

游戏驱动的快速迁移模型基座。在预训练底座上，通过数分钟自监督交互学习"动作在当前场景里会产生什么效果"，并以此为核心输出游戏实时指导信号。游戏是训练载体，不是最终产品。

> **现状（2026-07）**：当前运行主线是 **GRPO-Pixel 双塔线**（真 Minecraft / CraftGround）：
> 默认 v1 像素快塔，也可选择已接通 GRPO 的 v2 DINO+地图快塔；低频视觉语言模型提供子目标与像素指点，
> VPT 人类视频与教师分布提供动作先验，成对判官负责偏好精修。早期世界模型和 YOLOE 路线已退役，
> 仍有效的设计与实验结论统一收敛在 [knowledge/README.md](knowledge/README.md)。

---

## 快速安装

```bash
# 自动检测平台（Colab / 本机 / 服务器），按需加系统依赖
python install_env.py

# 或手动
pip install -e .          # 仅核心依赖
uv pip install -e .[dev]  # 加开发工具
```

当前主线的三个运行时依赖（CraftGround 环境 / Omni 慢塔 / Haiku 判官）与冒烟自检见
**[knowledge/README.md](knowledge/README.md)**。

---

## 设计原则（现行主线）

- **两个时间尺度**：快塔 20Hz 反应式动作（相机 mu-law 11-bin CE + 20 键 Bernoulli），
  慢塔 1Hz 输出文本子目标与像素指点；慢塔输出零阶保持，异步不阻塞快塔。
- **苦涩的教训**：不做人工领域先验（词表 / 手标 GT / 手写奖励代理）；
  利用大规模预训练的通用表征（DINO patch、VPT 人类视频）与可扩展的训练信号（BC 暖启动 + 判官排序精修）。
- **判官给序不给分**：相对优势由判官组内排序产生；手工统计量不进训练信号，里程碑只作汇报锚点。
- **特权信息只进训练侧**：raycast / env pose 只用于标定与评测，不进部署回路。
- **自标定**：分辨率 / FOV / 相机增益 / 步速是环境参数不是代码常量，开局探针实测（`net/calibration.py`）。

完整设计、证据、停止规则与运行约束见 [knowledge/README.md](knowledge/README.md)。

---

## 项目结构

```
blocks/            L1 算子库（attention/conv/encoder/decoder/distributions/dynamics/encodings/
                   quantization/regularization/sequence/mlp，I1–I8 已实现）
net/               网络组件（快塔 pixel_tower + 地图 map_io/ego_map + token_tower + DINO 前端 backbone/dino_tokenizer + 自标定 calibration）
  backbone.py      load_backbone（冻结 DINOv2/v3 HF 加载；mock 骨干见 tests/，经依赖注入）
  config.py        结构 schema（纯 dataclass，无 IO）
  pixel_tower.py   当前在跑的从零像素快塔（IMPALA 风格卷积干 + FiLM + 因果时序 + mu-law 相机头）
  map_io.py / fovea_twotower/ego_map.py  北锚定自我中心特征地图 IO（IPM/MapWriter/MapReader/AimPin）
  token_tower.py   goal-as-query cross-attention + UTF-8 字节语言 token（定稿未来结构，接线中）
train/             训练域：不同数据集的区分全压在这一层（数据契约 + 循环 + 装配）
  craftground/     当前运行时：grpo_pixel（GRPO 快塔）+ bc_vpt_warmstart（BC 暖启动）+ action_contract + env 系
  minecraft/       VPT 数据集域：vpt_action / vpt_dataset 数据契约（真数据下载见 tests/download_vpt_data.py）
  fovea_twotower/  grpo_harness（group_advantage）+ 判官对照（judge_exam 系/judge_train）+ 地图探针 + QLoRA 冒烟锚
  godot_meta_rl/   Godot RL 共享内存对接（vec_env：SB3 VecEnv 适配；用户 2026-07-10 裁决保留，未来可能启用）
utils/             通用基础设施：io（yaml 读取 + HF token）/ godot_rl（Godot 跨平台共享内存基础设施）
assets/godot_meta_rl/  Godot 引擎工程（C# Main.cs 编排 + GDScript 环境 + 场景），见其 README
tests/             unit/（当前运行时与定稿未来部件的单测，CPU/CUDA 可跑）+ 现行探针（probe_dino_aim / probe_vpt_calib / download_vpt_data）
knowledge/         设计文档（见下方文档索引）
runs/              下载数据 / checkpoints / 日志  [gitignored]
```

放置 / 写作 / 拆分规范见 [AGENTS.md](AGENTS.md) §8–§10。

---

## 环境

- **生产（训练）**：Linux + CUDA。核心依赖经 `pip install -e .` 安装；
  CraftGround 另需 Java 21 与 X 渲染（见 [knowledge/README.md](knowledge/README.md)）。
- **开发（测试 + net 前向）**：Windows + CUDA 同样可跑——Mamba 已弃用，不再有平台限制。
- DINOv3 权重受访问限制：需 HuggingFace token，经 Colab Secret（`HF_TOKEN`）或仓库根 `.env` 注入
  （`utils/io.py` 的 `get_hf_token`）。无 token 时使用开放权重 dinov2 预设。

---

## 训练与冒烟

```bash
# GRPO-Pixel 链路冒烟（groups=1 / ticks=120；需 CraftGround 环境）
python train/craftground/grpo_pixel.py --smoke

# VPT 人类视频 BC 暖启动（数据下载见 tests/download_vpt_data.py）
python -m train.craftground.bc_vpt_warmstart --help

# 用 BC checkpoint 暖启动 GRPO
python train/craftground/grpo_pixel.py --init-from runs/checkpoints/bc_vpt/best.pt
```

渲染路径选型（Xvfb / Xorg+RAW / ZEROCOPY）实测口径见
[knowledge/README.md](knowledge/README.md) §3。

---

## 测试

```bash
python -m pytest tests/unit/                       # 当前运行时与定稿未来部件单测（CPU/CUDA）
```

---

## 文档索引

| 文档 | 内容 |
|---|---|
| [AGENTS.md](AGENTS.md) | 助手约束与代码规范：I1–I8、生产纯净、SSOT、放置 / 写作 / 拆分合并 |
| [knowledge/README.md](knowledge/README.md) | 唯一知识库：现行架构、决策证据、停止规则、实现边界、安装与重放契约 |

Godot 子系统文档见 [assets/godot_meta_rl/README.md](assets/godot_meta_rl/README.md)。
