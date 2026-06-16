# 重构提议：建立清晰的 Blocks 与 Net 架构边界

您指出了一个非常核心且敏锐的架构痛点。

目前代码库中，神经网络相关的组件（尤其是各类型的 `nn.Module`）确实存在**“职责边界模糊、物理位置分散”**的问题：
1. **训练脚本中夹带网络**：`VPTBiasSidecar` 作为一个全局基率偏置网络层，被塞在 `train/vpt/distill_vpt.py` 内部。
2. **工具脚本中隐写模型**：`PoolHead`、`GridHead` 和 `PredOracle` 三个用于逆动力学上界分析的网络模块，被深埋在 `tools/oracle_idm.py` 工具脚本中。
3. **部分 Blocks 带有业务偏向**：部分用于诊断的薄探针或任务专用组件边界不清。

---

## 1. 重构的北极星原则：Blocks vs Net

为了实现高内聚、低耦合，我们应当在 `blocks/` 和 `net/` 之间划定一条不可逾越的边界：

```mermaid
graph TD
    subgraph blocks/ (原子无状态组件库)
        direction TB
        A[" pritives.py / yolo.py "]
        A1["数学/几何算子 (Warp, BEVSplat)"]
        A2["通用特征模块 (PreLNAttn, ConvGRU)"]
        A3["无状态编码器 (ContinuousTimeEncoding, PositionalEmbed)"]
        A4["防坍缩正则 (SIGReg)"]
    end

    subgraph net/ (有状态/业务相关网络模型)
        direction TB
        B[" world_model.py / heads.py / ... "]
        B1["MinecraftWorldModel (核心主干)"]
        B2["DecoderHeads / InverseDynamicsHead (Minecraft 契约解码)"]
        B3["VPTBiasSidecar (搬入: 动作规划偏置)"]
        B4["OracleHeads (搬入: PoolHead, GridHead)"]
    end

    net/ -->|组合/拼装| blocks/
```

### 🗃️ 积木库 `blocks/` 的职责
* **核心定位**：**纯粹无状态的、通用的、与具体业务数据集和动作维度解耦的原子层。**
* **准入规范**：
  * 🙅‍♂️ 禁止 import `net/` 中的任何模块。
  * 🙅‍♂️ 禁止硬编码与特定任务相关的常量（如 `ACTION_DIM = 22` 或特定任务 ID）。
  * 🙅‍♂️ 保持输入输出的纯数学几何语义（如 $C, H, W$ 特征通道映射）。

### 🗃️ 网络模型库 `net/` 的职责
* **核心定位**：**有状态的、与具体数据集/动作契约绑定、组合 blocks 原子组件的大型网络与解码头。**
* **准入规范**：
  * 🙆‍♂️ 负责拼装 `blocks/` 里的底层算子。
  * 🙆‍♂️ 承载具体的业务逻辑，与 `ACTION_DIM=22` 键盘/鼠标等契约布局直接绑定。
  * 🙆‍♂️ 包含所有训练、评估、诊断所需的读出头和 Sidecar 辅助层。

---

## 2. 具体迁移实施步骤

如果执行重构，我们可以按照以下三步方案进行无损迁移，确保 `train/` 和 `tools/` 保持 100% 纯净（无隐式网络定义）：

### 📂 步骤一：收拢蒸馏侧翼网络
* **操作**：将 `train/vpt/distill_vpt.py` 中的 `class VPTBiasSidecar` 物理迁移至 `net/heads.py`（或者在 `net/` 下新建 `net/sidecar.py`）。
* **修改引用**：
  * 修改 `train/vpt/distill_vpt.py` 中的导入语句：
    `from net.heads import VPTBiasSidecar`（或对应新文件）。

### 📂 步骤二：收拢 Oracle 诊断网络
* **操作**：将 `tools/oracle_idm.py` 中的 `PoolHead`、`GridHead` 和 `PredOracle` 迁入 `net/` 目录。我们可以在 `net/` 下新建 `net/oracle_heads.py`（或者直接并入 `net/world_probe.py`）。
* **修改引用**：
  * 修改 `tools/oracle_idm.py` 中的导入语句。
  * 这样所有逆动力学上限估计的诊断网络在 `net/` 目录下集中管控。

### 📂 步骤三：运行冒烟与集成测试
* **操作**：执行自动化测试：
  ```bash
  pytest tests/unit/
  ```
  校验所有 `import` 引用已被正确修正，前向/反向传播逻辑完好无损。
