# TAO-Not-42

游戏驱动的快速迁移模型基座。在预训练底座上,通过数分钟自监督交互学会"动作在当前场景里
有什么效果",并以此为核心输出游戏实时指导信号。游戏是练兵场,不是终产品。

> ⚠️ **重构中(2026-06,统一世界基座清白重设计)**:原 **Minecraft Δz-JEPA 世界模型**
> (`net/world_model.py` 等)与并行的 **RSSM+后继特征切片**(`net/rssm.py`)**已删除退役**,
> 仓库正切换到一个**清白重设计、跨域共享权重的统一世界基座**(从 `blocks/` 算子库组装)。
> 设计意图见 [knowledge/mental_world.md](knowledge/mental_world.md) 的退役公告;新基座设计文
> `knowledge/world_foundation.md` 与 `net/` 实现待构建期补入。**本文下方"项目结构 / 训练 / 测试 /
> 诊断工具"小节描述的是已删旧管线,暂作历史,待新基座落地后重写。** 当前可运行:`blocks/`、
> `net/backbone.py`、`train/minecraft/`(数据契约 + VPT teacher)、Godot RL 子系统、`tests/`。

当前北极星仍是"看视频掌握玩法"的快速迁移底座(见 mental_world §6);Δz-JEPA 是其已退役的第一版载体。

---

## 设计原则(一句话)

- **JEPA 潜空间预测**:不解码回像素,预测潜表征**增量** Δz;persistence(预测 0)= 1.0 基线。
- **冻结视觉骨干 + EMA 目标**:DINOv3 ViT-S/16 冻结,目标编码器是在线权重的 EMA 副本(平稳靶)。
- **逆动力学接地可控闸 c**:从潜变化反推动作,把"哪些变化由动作引起"压进 c。
- **世界模型退到训练期**:动力学预测用于自监督与想象式优化,推理期不在控制环里跑 rollout。
- 框架是 **Transformer**(已弃用 Mamba:核心状态改为有限抽象潜向量后,逐像素 SSM 的前提消失)。

---

## 项目结构

```
blocks/            L1 算子库(PreLNAttn/GatedResidual/SIGReg/ContinuousTimeEncoding/...,I1–I8 焊进实现)
net/               网络组件
  world_model.py   MinecraftWorldModel —— Δz-JEPA 活模型(顶层装配)
  slots.py         SlotBinder / SlotCompetitiveAttn(实体槽绑定)
  backbone.py      load_backbone(冻结 DINOv2/v3 HF 加载;mock 骨干见 tests/,经依赖注入)
  heads.py         DecoderHeads(未来动作规划)/ InverseDynamicsHead(逆动力学)
  world_probe.py   世界探针
  vpt_lib/         vendored OpenAI VPT(第三方,见 NOTICE;不受代码规范约束)
train/             训练域:不同数据集的区分全压在这一层(数据契约 + 循环 + 装配)
  minecraft/         VPT/BASALT 数据集域(旧 train_minecraft/losses/eval 已删,训练循环待新基座补)
    vpt_action.py    动作 ↔ 张量契约 + mu-law 相机分箱(SSOT)
    vpt_dataset.py   VPTStreamDataset(流式 uint8 加载 / 可变帧跨度采样)
    task_text.py     冻结句向量任务条件
    vpt_teacher.py   minerl-free VPT teacher(软 KL 蒸馏的边缘化适配器)
  godot_meta_rl/     Godot 40 环境 RL 训练/诊断/协议测试(聚光灯瞄准独有,不可复用)
    vec_env.py          GodotVecEnv(SB3 VecEnv 适配)/ RolloutProgress
    train_ppo*.py       锁步 / 线程异步 / 双进程 三种 PPO 执行器
    smoke.py / diag_montage.py / async_min.py / test_*.py  冒烟·诊断·协议测试
utils/             通用基础设施(data/geometry/losses/matching/nn/probes/visualization/hf_token)
  godot_rl/          Godot RL 跨平台基础设施:shared_mem_env(文件后端共享内存+轮询握手驱动)/
                     launch(启停 Godot)/ ppo_factory(build_model/make_buffer/buffer 搬运)
assets/godot_meta_rl/  Godot 引擎工程(C# Main.cs 编排 + GDScript 环境 + 场景),Python 侧见上
tests/             unit/(几何/损失/SIGReg,CPU 可跑)+ 离线脚本(download_sample_data 合成数据 /
                   test_dinov3_hf 骨干冒烟)
knowledge/         设计文档:mental_world(愿景)/ code_conventions(代码规范)
runs/              下载数据 / checkpoints / 日志  [gitignored]
```

详细放置 / 写作 / 拆分规范见 [knowledge/code_conventions.md](knowledge/code_conventions.md)。

---

## 环境

- **生产(训练)**:Linux + CUDA。依赖见 [requirements.txt](requirements.txt)(torch / transformers /
  opencv / numpy / wandb 等)。
- **开发(测试 + net 前向)**:Windows + CUDA 同样可跑——Mamba 已弃用,不再有平台门槛。
- DINOv3 权重 **gated**:需 HuggingFace token,经 Colab Secret(`HF_TOKEN`)或仓库根 `.env` 注入
  (`utils/hf_token.py` 双重加载)。无 token 用 `--config configs/minecraft/dinov2.yaml`(开放权重);
  离线管线冒烟见 `tests/integration/`。

```bash
pip install -r requirements.txt
```

---

## 训练

```bash
# 真实训练(DINOv3 骨干 = 默认 base 预设,需 HF token + VPT 数据)
python train/minecraft/train_minecraft.py \
    --data_dir runs/vpt_sample --holdout_dir runs/vpt_holdout \
    --config configs/minecraft/base.yaml --img_size 128 --batch 128 --epochs 300 --device cuda

# 无 HF token 时用开放权重 dinov2(首次下载后本地缓存)
python train/minecraft/train_minecraft.py --data_dir runs/vpt_sample \
    --config configs/minecraft/dinov2.yaml --epochs 1

# 模型结构(d/N/K/J、骨干、binder、dynamics、heads、ξ)全在 yaml 预设里改;CLI 只剩训练/数据旋钮
python train/minecraft/train_minecraft.py --help
```

Colab 端数据准备与一键训练见 `colab_demo.ipynb`(gitignored)。

---

## 测试

```bash
python -m pytest tests/unit/          # 几何 / 损失 / SIGReg(CPU)
python -m pytest tests/integration/   # 活模型离线冒烟(mock 骨干,前向+反向+EMA)
```

---

## 诊断工具

```bash
# 逆动力学上界:冻结特征里到底能读出多少动作(钉死瓶颈在编码还是读出)
python tools/oracle_idm.py --checkpoint runs/mc_ckpt/best.pt
```

---

## 文档索引

| 文档 | 内容 |
|---|---|
| [AGENTS.md](AGENTS.md) | 助手约束:数值不变量 I1–I8、生产纯净、SSOT、写作纪律 |
| [knowledge/code_conventions.md](knowledge/code_conventions.md) | 代码组织规范:放置 / 写作 / 拆分合并 |
| [knowledge/mental_world.md](knowledge/mental_world.md) | 脑内世界设计愿景(宏观架构与算法意图) |
| [knowledge/claims_and_scope.md](knowledge/claims_and_scope.md) | 期望与现实校准:当前能诚实主张的命题、主张边界、哪条线朝元愿望走 |
| [knowledge/world_model_landscape.md](knowledge/world_model_landscape.md) | 世界模型外部版图调研与设计对照 |
