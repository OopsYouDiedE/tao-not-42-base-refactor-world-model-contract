# 两步式世界模型重构与清理完成说明 (walkthrough.md)

本重构工作已根据用户关于“两步式”架构的设计指示，彻底清除了废弃的 Slot、随机隐变量 $\xi$、VPT 动作蒸馏以及控制重映射相关的残留代码与文件。同时修改了核心世界模型前向逻辑，实现了第一步离散动作 Token 自回归预测，与第二步基于 Cross-Attention 的当前帧 patch 潜向量重建。

---

## 1. 物理删除的废弃模块 (Deleted Files)

以下模块在两步式模型中不再需要，已进行彻底物理删除：

* [domains/minecraft/control_remap.py](file:///c:/Users/zznZZ/Desktop/tao-not-42-base-refactor-world-model-contract/domains/minecraft/control_remap.py)：物理控制动作重映射。
* [net/oracle_heads.py](file:///c:/Users/zznZZ/Desktop/tao-not-42-base-refactor-world-model-contract/net/oracle_heads.py)：原 Oracle 评估专用头。
* [net/slots.py](file:///c:/Users/zznZZ/Desktop/tao-not-42-base-refactor-world-model-contract/net/slots.py)：Slots 绑定与竞争注意力机制。
* [train/minecraft/minecraft_viz.py](file:///c:/Users/zznZZ/Desktop/tao-not-42-base-refactor-world-model-contract/train/minecraft/minecraft_viz.py)：可视化面板。
* [train/vpt/distill_vpt.py](file:///c:/Users/zznZZ/Desktop/tao-not-42-base-refactor-world-model-contract/train/vpt/distill_vpt.py)：VPT 行为克隆蒸馏管线。

---

## 2. 核心架构修改 (Architectural Refactoring)

### 2.1 世界模型与预测/重建头
* **修改文件**：[net/world_model.py](file:///c:/Users/zznZZ/Desktop/tao-not-42-base-refactor-world-model-contract/net/world_model.py)
  * 去除 EMA 目标网络、随机隐变量 $\xi$ 以及 Slots 的所有逻辑。
  * **第一步**：对上一帧 patch 特征进行平均池化，结合历史状态/动作时序送入自回归 Transformer，预测出下一步的动作词表 Token Logits。
  * **第二步**：利用 GT（或预测的）动作 Token Embedding 作为 Key/Value，与上一帧的细粒度 patch 潜向量进行 Cross-Attention 重构，输出重建的当前帧 patch 潜向量 `z_recon`。
* **修改文件**：[net/heads.py](file:///c:/Users/zznZZ/Desktop/tao-not-42-base-refactor-world-model-contract/net/heads.py)
  * 移除了无用的逆动力学、多阶段动作预测头等，引入极简的自回归 Token 预测头 `ActionVocabHead`。

### 2.2 损失函数与训练流
* **修改文件**：[train/minecraft/losses.py](file:///c:/Users/zznZZ/Desktop/tao-not-42-base-refactor-world-model-contract/train/minecraft/losses.py)
  * 移除了 KL 散度与槽重叠度损失，简化为 CrossEntropy 分类损失 `vocab_pred_loss` 与 MSE 特征重建损失 `z_recon_loss`。
* **修改文件**：[train/minecraft/train_minecraft.py](file:///c:/Users/zznZZ/Desktop/tao-not-42-base-refactor-world-model-contract/train/minecraft/train_minecraft.py) 与 [train/minecraft/eval.py](file:///c:/Users/zznZZ/Desktop/tao-not-42-base-refactor-world-model-contract/train/minecraft/eval.py)
  * 接入 `ActionTokenizer` 提取真实动作序列对应的离散 Token ID 作为监督信号，重构了截断 BPTT 训练流程与评价指标。

### 2.3 配置与测试更新
* **修改文件**：[configs/minecraft/tiny.yaml](file:///c:/Users/zznZZ/Desktop/tao-not-42-base-refactor-world-model-contract/configs/minecraft/tiny.yaml) & [configs/minecraft/base.yaml](file:///c:/Users/zznZZ/Desktop/tao-not-42-base-refactor-world-model-contract/configs/minecraft/base.yaml)
  * 移除 `slots`、`xi`、`encoder` 相关的冗余参数。
* **修改文件**：[tests/integration/test_pipeline.py](file:///c:/Users/zznZZ/Desktop/tao-not-42-base-refactor-world-model-contract/tests/integration/test_pipeline.py) & [tests/unit/test_config.py](file:///c:/Users/zznZZ/Desktop/tao-not-42-base-refactor-world-model-contract/tests/unit/test_config.py)
  * 更新测试以对齐两步式架构和简化的配置。

---

## 3. 验证结果 (Validation Results)

在重构后，我们执行了项目全套单元测试与集成测试：
```bash
python -m pytest
```
**结果**：
`20 passed, 9 warnings in 5.69s`

所有测试顺利通过，确认无 NaN 梯度，前向与反向传播完全正常。

**Git Commit**：
已执行 `git commit`：
```
[main b5b1282] refactor: Reconstruct world model to two-step discrete token rollout and cross-attn feature reconstruction
 15 files changed, 399 insertions(+), 2972 deletions(-)
 delete mode 100644 domains/minecraft/control_remap.py
 delete mode 100644 net/oracle_heads.py
 delete mode 100644 net/slots.py
 delete mode 100644 train/minecraft/minecraft_viz.py
 delete mode 100644 train/vpt/distill_vpt.py
```
共物理精简代码 2573 行。
