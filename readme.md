# TAO-Not-42

游戏驱动的快速迁移模型基座。在预训练好的底座上,通过数分钟自监督交互学会"动作在当前场景里有什么效果",并以此为核心输出游戏实时指导信号。

游戏是练兵场,不是终产品。

---

## 设计原则

- **跳蛛网络(System-1)**:推理期只跑一个紧凑的、游戏嵌入条件化的反应式策略,无 rollout,无规划。
- **Prompt Tuning**:每个游戏只有一个私有参数——一个可学嵌入向量 `g`。整个网络权重共享,适应新游戏只优化这一个向量(几百步 ≈ 数分钟)。
- **世界模型退到训练期**:动力学预测(DynamicsStep)仅在训练期用于自监督预训练和想象式策略优化(System-2),推理期不在控制环里跑 rollout。
- **Mamba 时序**:运动对齐(LocalCorr → warp)后再跑 Mamba,避免逐像素 SSM 无法建立运动对应的问题。

详见 [Plan.md](Plan.md)。

---

## 项目结构

```
blocks/          L1 primitive 积木库(LocalCorr/Warp/FiLM/Mamba接口等)
net/             网络组件
  encoder.py     Encoder + GameEmbed + PresetBank  (输入侧)
  heads.py       所有输出头:几何/追踪/逆动力学/ActionAdapter/Policy
  dynamics.py    DynamicsStep  (有界平流 + 随机事件)
  agent.py       顶层装配
  yoloe.py       YOLOE 检测栈(substrate 门控可选头)
  legacy.py      待删旧算子(canary 通过后物理删除)
train/           训练基础设施
  dataset.py     数据源:NitroGenSource + ProcgenSource + MixedSource
  trainer.py     预训练主循环(Reptile + 自监督动力学 + 逆动力学)
  train.py       入口
tests/           测试(含 Mamba Mock,CPU 可跑)
knowledge/       设计文档(宏观架构与算法意图,不描述实现细节)
runs/            下载数据 / checkpoints / 日志  [gitignored]
```

---

## 环境要求

- **生产(训练)**:Linux + CUDA + `mamba_ssm` + `procgen` + `datasets`
- **开发(仅跑测试)**:Windows + CUDA;`tests/` 里的 Mamba Mock 自动注入

```bash
pip install torch torchvision mamba-ssm procgen datasets
```

---

## 预训练

```bash
# 单游戏 Procgen 快速验证
python -m train.train --sources procgen --procgen_games coinrun --steps 50000

# NitroGen 多游戏预训练(需生产机)
python -m train.train --sources nitrogen procgen --run_name pretrain_mixed

# 查看可选参数
python -m train.train --help
```

checkpoint 保存在 `runs/checkpoints/<run_name>/`。

---

## 测试

```bash
# CPU 可跑:blocks + net 单元测试(Mamba 由 conftest.py Mock)
pytest tests/unit/

# 集成测试
pytest tests/integration/
```

---

## 文档索引

| 文档 | 内容 |
|---|---|
| [Plan.md](Plan.md) | 项目路线图:数据集候选、模型架构、验收指标、推进顺序 |
| [AGENTS.md](AGENTS.md) | 助手约束与开发规范(I1–I8、SSOT、测试纪律) |
| [knowledge/world_model_interface.md](knowledge/world_model_interface.md) | 数据契约 v1.0(SSOT) |
| [knowledge/net_blocks.md](knowledge/net_blocks.md) | L1 primitive 积木库规格 |
| [knowledge/yolo.md](knowledge/yolo.md) | YOLOE 检测栈参考 |
| [knowledge/map_representation.md](knowledge/map_representation.md) | 地图子系统(未来阶段) |
