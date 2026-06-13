# AI 助手（Antigravity）核心修改记录

本文件由 AI 助手维护，记录近期对核心架构、Bug 修复和训练管线的重要修改。

## [2026-06-14] 世界模型核心 Bug 修复与架构升级

### 1. 修复核心动作契约（VPT Keys）索引错位 Bug
**背景**：在 `oracle_idm.py`（用于评估和可视化逆动力学）与 `vpt_action.py`、`vpt_dataset.py` 之间存在隐蔽的索引绑定矛盾。训练时模型将“左移”视为索引 3，但评估可视化时却认为索引 3 是“后退”。
**修改**：
- 以 `train/minecraft/vpt_dataset.py` 中 `VPT_KEYS` 定义的真实训练数据契约为单一事实来源（SSOT）。
- 重构 `train/minecraft/vpt_action.py` 的 `VPT_KEYS` 列表和 `encode_vpt_jsonl` 的字典映射，使其与训练集对齐。
- 更新 `tools/oracle_idm.py` 的 `RAW2IDX` 字典映射，彻底解决 Inv-dyn（逆动力学）可视化乱码和 Loss 评估张冠李戴的问题。

### 2. 解决对象槽（Slot）多槽位空间注意力坍缩问题
**背景**：在纯 JEPA（无像素重建损失）训练范式下，`SlotCompetitiveAttn` 的 `softmax` 竞争机制失效，导致所有 16 个 Slot 寻找捷径，提取同一块最显著物体（如移动的草地）的特征，完全丧失了对象分离能力。虽然 `SIGReg` 能防止全图特征坍缩为常量，但无法解决多槽位在同一张图内互相重复的内部坍缩。
**修改**：
- 在 `net/tao_not_42.py` 的 `SlotCompetitiveAttn` 内部，针对空间注意力聚合权重 `w`（维度 `[B, h, N, M]`）施加**施密特正交化（Gram-Schmidt Orthogonalization, GSO）**。
- 这强制不同 Slot 在图像 Patch 维度的注意力分布必须相互正交（互不重叠），形成类似于 Matching Pursuit 的残差槽特征空间。
- **为何在注意力权重 `w` 上正交化？** 实验和逻辑推演证实，如果在特征空间（`delta_Z`）上正交化，会导致模型无法同时追踪画面内两个同类长相（语义特征共线）的物体。在注意力图 `w` 上正交化，不仅完美剥离了空间焦点，还保留了语义追踪的一致性。
- 修改了 `knowledge/mental_world.md` 对应文档。

*注：相关代码操作均符合 `I1`（分母 clamp 1e-4）和 `I4`（投影强化为 fp32）等数值安全性硬约束。*

> **[2026-06-14 复盘修正 / Claude]** 上述第 2 点的硬 GSO 已**撤销**：在前向里正交化聚合权重 `w` 会破坏 `out = w·v` 的加权平均语义（行不再非负、和≠1），并按 slot 序饿死后排槽（把"都看同一物体"换成"首槽看、后排饿死"）。改为**保留**竞争 softmax + 把"槽间别盯同一块"写成**软惩罚**：`slot_diversity_loss`（`--beta_div`，惩罚竞争注意力图的成对重叠），前向 `w` 仍是合法分布。空间仍在注意力图（与原判断一致），只是从"硬改前向"换成"进损失"。详见 `knowledge/mental_world.md` 与 `train/minecraft/train_minecraft.py`。
