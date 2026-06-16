# 重构提议：建立清晰的 Blocks 与 Net 架构边界

您的意见非常具有前瞻性！

在将网络组件收拢后，**`blocks/` 积木库内部不应该是一个臃肿的大乱炖文件（如现有的 `primitives.py`），而应该按照“职责内容相似、功能在逻辑上可以替换”的原则，拆分到不同的专有子模块中。**

这不仅能提高代码的可读性，更重要的是，**当我们需要尝试不同的技术方案时（例如将绝对位置编码替换为傅里叶坐标 PE），我们可以直接在同一模块下做无缝平替。**

---

## 1. 重构的北极星原则：Blocks vs Net

为了实现高内聚、低耦合，我们应当在 `blocks/` and `net/` 之间划定一条不可逾越的边界：

```mermaid
graph TD
    subgraph blocks/ (按相似性与平替性细分)
        direction TB
        B_sp["spatial.py (空间几何与投影)"]
        B_sim["similarity.py (关联与流估计)"]
        B_enc["encodings.py (时空位置编码)"]
        B_dyn["dynamics.py (动力学与门控)"]
        B_reg["regularization.py (正则与有界采样)"]
        B_attn["attention.py (多头注意力)"]
        B_yolo["yolo.py (YOLO 骨干积木)"]
    end

    subgraph net/ (具体业务相关的有状态模型)
        direction TB
        N1["world_model.py (世界模型)"]
        N2["heads.py (业务解码头与 VPTBiasSidecar)"]
        N3["oracle_heads.py (评估专用诊断头)"]
    end

    net/ -->|组合/拼装/平替| blocks/
```

---

## 2. `blocks/` 内部子模块细划方案（相似平替原则）

原 `blocks/primitives.py` 内部的大量无状态算子将按照其**“负责领域与可替换性”**被精细化拆解到以下专有文件中：

### 📂 2.1 `blocks/spatial.py` (空间几何与投影)
* **核心内容**：负责处理图像/特征图在 $2\mathrm{D}/3\mathrm{D}$ 空间上的仿射变换、位移重采样与 BEV 投影。
* **包含组件**：
  * `Warp` (基于光流的局部重采样)
  * `GlobalTransformApply` (屏幕仿射变换)
  * `BEVSplat` (3D相机到俯视 BEV 栅格投影)
  * `rot6d_to_matrix` / `make_4x4` (位姿三维旋转齐次矩阵辅助算子)
* **平替语义**：处理相机/物体移动在画面上产生的物理几何变动。

### 📂 2.2 `blocks/similarity.py` (关联与流场估计)
* **核心内容**：负责处理图块之间的相似度对比、稠密匹配与位移估计。
* **包含组件**：
  * `LocalCorr` (局部通道归一化余弦互相关)
  * `SoftArgmaxFlow` (相关性热图到期望位移流)
  * `box_iou` (2D 边界框 IoU/GIoU 相似度计算)
* **平替语义**：寻找不同时间帧或不同特征图之间的空间像素/局部对应特征。

### 📂 2.3 `blocks/encodings.py` (时空与位置编码)
* **核心内容**：将连续的时间戳、空间坐标或裁剪尺度，通过频域映射成高维正交特征向量。
* **包含组件**：
  * `PositionalEmbed` (2D正弦位置位置 PE)
  * `ContinuousTimeEncoding` (以帧为单位的可变时间 $\Delta t$ 连续时间 PE)
  * `SpatialPosEmbed` (局部注视裁剪坐标及尺度的傅里叶位置 PE)
* **平替语义**：**时空自定位编码的备选池**。当需要改进位置敏感度时，模型可以在此文件中无缝替换不同的编码基底。

### 📂 2.4 `blocks/dynamics.py` (动力学与门控更新)
* **核心内容**：处理时序推演中的记忆门控、渐进状态改写及条件信息融入。
* **包含组件**：
  * `ConvGRUCell` (2D 卷积门控循环单元)
  * `GatedResidual` (参数化限制增益残差门，符合不变量 $I_5$)
  * `FiLM` (条件特征线性仿射调制器)
  * `Accumulator` (精确数值累加器)
  * `DiscreteRouter` (直通可微硬选择路由器)
* **平替语义**：决定跨时间步状态如何被门控改写，是多步盲滚（rollout）时防止数值发散的盾牌。

### 📂 2.5 `blocks/regularization.py` (正则化与有界采样)
* **核心内容**：提供概率分布采样、有界激活限幅，以及维护表征空间各向同性的正则化。
* **包含组件**：
  * `SIGReg` (經驗高斯特征 Sliced 正则层，防坍缩了承重墙)
  * `StochLatent` (重参数化高斯与 Categorical 随机采样层)
  * `BoundedActivation` (执行指数/双曲正切等有界激活，符合不变量 $I_3$)
* **平替语义**：约束特征表征形态，防止 JEPA 表征坍缩或预测输出越界。

### 📂 2.6 `blocks/attention.py` (通用注意力层)
* **核心内容**：通用自注意力、交叉注意力及特征重建模块。
* **包含组件**：
  * `PreLNAttn` (预层归一化多头注意力)
  * `ProtoDecode` (基于 einsum 的全局注意力原型重建层)

### 📂 2.7 `blocks/yolo.py` (YOLO 特征检测积木)
* **核心内容**：卷积骨干与多尺度池化、位置敏感通道注意力层，已高度归类。

---

## 3. 具体迁移与重构方案

### 📂 步骤一：`blocks/` 目录的精细化拆分
1. 将原 `blocks/primitives.py` 物理删除。
2. 按照上面第 2 节的规划，拆建 6 个新文件：`spatial.py`, `similarity.py`, `encodings.py`, `dynamics.py`, `regularization.py`, `attention.py`。

### 📂 步骤二：收拢 `net/` 下的业务模型
1. **收拢蒸馏侧翼**：将 `train/vpt/distill_vpt.py` 中的 `VPTBiasSidecar` 迁入 `net/heads.py`。
2. **收拢 Oracle 诊断头**：在 `net/` 下新建 `net/oracle_heads.py`，迁入 `tools/oracle_idm.py` 中的 `PoolHead`、`GridHead` 和 `PredOracle`。

### 📂 步骤三：批量修正 `import` 引用与冒烟测试
1. 全局修正各文件的引用路径（例如 `from blocks.encodings import ContinuousTimeEncoding`）。
2. 执行 `pytest tests/` 确保所有前向、反向及测试全数通过，确保零结构损坏。
