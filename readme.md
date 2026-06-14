# TAO-Not-42

游戏驱动的快速迁移模型基座。在预训练底座上,通过数分钟自监督交互学会"动作在当前场景里
有什么效果",并以此为核心输出游戏实时指导信号。游戏是练兵场,不是终产品。

当前活跃管线是 **Minecraft Δz-JEPA 世界模型**:在离线 VPT 录像(画面 + 动作)上做自监督,
学世界动力学。设计愿景见 [knowledge/mental_world.md](knowledge/mental_world.md)。

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
domains/minecraft/ 数据契约与领域逻辑(非训练循环)
  vpt_action.py    动作 ↔ 张量契约 + mu-law 相机分箱(SSOT)
  vpt_dataset.py   VPTStreamDataset(流式加载)
  control_remap.py 逐 episode 控制重映射(in-context 看视频掌握玩法)
  task_text.py     冻结句向量任务条件
train/             训练基础设施(只放循环 + 装配)
  minecraft/
    train_minecraft.py  入口 / CLI / 主循环(_run_sequence/train_epoch/main)
    losses.py           5 个损失函数
    eval.py             evaluate / rollout_probe(离线诊断)
    _seq.py             roll_hist / _to_float_img(train↔eval 共用低层助手)
    minecraft_viz.py    训练面板可视化
  vpt/
    distill_vpt.py      VPT teacher 软 KL 蒸馏
utils/             通用基础设施(data/geometry/losses/matching/nn/probes/visualization/hf_token)
tools/             离线脚本:oracle_idm(逆动力学上界诊断)/ download_sample_data / vpt_teacher
tests/             unit/(几何/损失/SIGReg,CPU 可跑)+ integration/(活模型离线冒烟,DI 注入 mock 骨干)
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
  (`utils/hf_token.py` 双重加载)。无 token 用 `--encoder dinov2`(开放权重);离线管线冒烟见 `tests/integration/`。

```bash
pip install -r requirements.txt
```

---

## 训练

```bash
# 真实训练(DINOv3 骨干,需 HF token + VPT 数据)
python train/minecraft/train_minecraft.py \
    --data_dir runs/vpt_sample --holdout_dir runs/vpt_holdout \
    --encoder dinov3 --img_size 128 --batch 128 --epochs 300 --device cuda

# 无 HF token 时用开放权重 dinov2(首次下载后本地缓存)
python train/minecraft/train_minecraft.py --data_dir runs/vpt_sample --encoder dinov2 --epochs 1

# 全部参数
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
