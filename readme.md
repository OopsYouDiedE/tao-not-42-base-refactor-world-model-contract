# TAO-Not-42

游戏驱动的快速迁移模型基座。在预训练底座上，通过数分钟自监督交互学习"动作在当前场景里会产生什么效果"，并以此为核心输出游戏实时指导信号。游戏是训练载体，不是最终产品。

> **重构中（2026-06，统一世界基座重设计）**：原 **Minecraft Δz-JEPA 世界模型**（`net/world_model.py` 等）
> 与并行的 **RSSM + 后继特征切片**（`net/rssm.py`）已删除退役。当前仓库的世界模型从 `blocks/` 算子库重新组装：
> `net/dreamerv3/`（可训练，已在 Crafter 上跑通）、`net/dreamer4/`（仅构建）。跨域共享权重的统一基座仍在设计中。
> 设计意图见 [knowledge/mental_world.md](knowledge/mental_world.md)，Dreamer 系实现见 [knowledge/dreamer.md](knowledge/dreamer.md)。

当前目标仍是"看视频掌握玩法"的快速迁移底座（见 [knowledge/mental_world.md](knowledge/mental_world.md) §6）；Δz-JEPA 是已退役的第一版实现。

---

## 设计原则

- **JEPA 潜空间预测**：不解码回像素，预测潜表征**增量** Δz；persistence（预测 0）= 1.0 基线。
- **冻结视觉骨干 + EMA 目标**：DINOv3 ViT-S/16 冻结，目标编码器是在线权重的 EMA 副本（稳定训练目标）。
- **逆动力学接地可控闸 c**：从潜变化反推动作，把"哪些变化由动作引起"压进 c。
- **世界模型退到训练期**：动力学预测用于自监督与想象式优化，推理期不在控制环里跑 rollout。
- 框架采用 **Transformer**（已弃用 Mamba：核心状态改为有限抽象潜向量后，逐像素 SSM 的适用前提消失）。

---

## 项目结构

```
blocks/            L1 算子库（attention/conv/encoder/decoder/distributions/dynamics/encodings/
                   quantization/regularization/sequence/mlp，I1–I8 已实现）
net/               网络组件
  backbone.py      load_backbone（冻结 DINOv2/v3 HF 加载；mock 骨干见 tests/，经依赖注入）
  config.py        结构 schema（纯 dataclass，无 IO）
  dreamerv3/       DreamerV3 世界模型（从 blocks 重建：RSSM + 编解码 + 想象 actor-critic + 稀疏 planner；可训练）
  dreamer4/        Dreamer4 时空 Transformer 世界模型（从 blocks 组装：tokenizer + shortcut forcing；仅构建）
  ppo_ad/          Crafter PPO + Achievement Distillation actor-critic
  vpt_lib/         vendored OpenAI VPT（第三方，见 NOTICE；不受代码规范约束）
train/             训练域：不同数据集的区分全压在这一层（数据契约 + 循环 + 装配）
  crafter/         Crafter 域：env / 回放 / PPO+AD（train_ppo_ad）/ DreamerV3（train_dreamerv3）/ goal / planner
  minecraft/       VPT 数据集域：vpt_action / vpt_dataset / task_text 数据契约（训练循环待新基座补）
  godot_meta_rl/   Godot RL 共享内存对接（vec_env：SB3 VecEnv 适配）
utils/             通用基础设施：io（yaml 读取 + HF token）/ godot_rl（Godot 跨平台共享内存基础设施）
assets/godot_meta_rl/  Godot 引擎工程（C# Main.cs 编排 + GDScript 环境 + 场景），见其 README
tests/             unit/（SIGReg / 空间位置编码，CPU 可跑）+ integration/（test_dreamer_build）
knowledge/         设计文档（见下方文档索引）
runs/              下载数据 / checkpoints / 日志  [gitignored]
```

放置 / 写作 / 拆分规范见 [AGENTS.md](AGENTS.md) §8–§10。

---

## 环境

- **生产（训练）**：Linux + CUDA。依赖见 [requirements.txt](requirements.txt)（torch / transformers /
  opencv / numpy / wandb 等）；Crafter 训练另需 `pip install crafter`。
- **开发（测试 + net 前向）**：Windows + CUDA 同样可跑——Mamba 已弃用，不再有平台限制。
- DINOv3 权重受访问限制：需 HuggingFace token，经 Colab Secret（`HF_TOKEN`）或仓库根 `.env` 注入
  （`utils/io.py` 的 `get_hf_token`）。无 token 时使用开放权重 dinov2 预设。

```bash
pip install -r requirements.txt
```

---

## 训练

当前可训练的世界模型是 Crafter 上的 DreamerV3，操作步骤与调参见 [knowledge/dreamer.md](knowledge/dreamer.md)。

```bash
# Crafter DreamerV3（冒烟：tiny，约 4k 步）
python -m train.crafter.train_dreamerv3 --size tiny --total-steps 4000 \
    --prefill 500 --run-dir runs/crafter_smoke

# Crafter DreamerV3（正式：small，后台 + 行缓冲日志）
nohup python -m train.crafter.train_dreamerv3 --size small --total-steps 200000 \
    --run-dir runs/crafter_dreamerv3 > runs/crafter_dreamerv3/train.log 2>&1 &

# Crafter PPO + Achievement Distillation
python -m train.crafter.train_ppo_ad --help
```

---

## 测试

```bash
python -m pytest tests/unit/                       # SIGReg / 空间位置编码（CPU）
python -m pytest tests/integration/                # DreamerV3 构建 + CPU 前向/反向冒烟
```

---

## 文档索引

| 文档 | 内容 |
|---|---|
| [AGENTS.md](AGENTS.md) | 助手约束与代码规范：I1–I8、生产纯净、SSOT、放置 / 写作 / 拆分合并 |
| [knowledge/code_analysis.md](knowledge/code_analysis.md) | 精确到函数的代码结构与方法说明 |
| [knowledge/mental_world.md](knowledge/mental_world.md) | 设计愿景与诚实边界（宏观架构、算法意图、当前能主张什么） |
| [knowledge/dreamer.md](knowledge/dreamer.md) | Dreamer 系（DreamerV3 + Dreamer4）设计与 Crafter 训练手册 |
| [knowledge/world_model_landscape.md](knowledge/world_model_landscape.md) | 世界模型外部版图调研与设计对照 |

Godot 子系统文档见 [assets/godot_meta_rl/README.md](assets/godot_meta_rl/README.md)。
