# Net-Like Classes and Functions Statistics

本文件全面统计并分类了当前代码库中**所有具有“网络层（Net/Module）”或“神经网络计算（前向提取、编码、权重更新）”特征的类和函数**。

这不仅包括继承自 `nn.Module` 的显式网络模型与子层，也包括那些没有继承自 `nn.Module`，但在逻辑功能上扮演神经网络特征提取、位置编码、参数平滑更新、数据归一化和动作分箱编解码的核心网络计算函数。

---

## 1. 继承自 `nn.Module` 的核心网络组件 (非 blocks 库部分)

以下类全部继承自 `nn.Module`（或在内置库中隐式实现），属于显式的神经网络模型与网络子层：

### 1.1 核心网络架构层 (`net/` 目录)
* **`MinecraftWorldModel`** (`net/world_model.py`)
  * **职责**：序列对齐的后果结构世界模型主干。冻结 DINO → 可训练 adapter 编码因子化潜
    `z=(z_rev 有界连续, z_inv 随机潜+KL 瓶颈)`;把所有帧的 patch 加屏幕坐标 sine2d + 帧单位
    时间编码摊平成**时空 token 集合**,与未来 query token 一次性送入 Transformer 核做
    **掩码未来帧预测**;因子化输出 `ẑ_rev=z_rev+Σc·G_j`、`ẑ_inv=z_inv+𝒟.decode`,`do(null)`
    旁路给反事实效应 e。目标编码取 **EMA 教师 `target_adapter` + stop-grad**(JEPA,I8)。
* **`_Adapter`** (`net/world_model.py`)
  * **职责**:冻结骨干之上的可训练编码器(PreLNAttn 自注意 + GatedResidual MLP)→ 两路头
    (BoundedActivation('flow') 的 z_rev / StochLatent 的 z_inv)。online 与 EMA 教师同构。
* **`EffectTokenizer`** (`net/effect_tokenizer.py`)
  * **职责**：不可逆事件词表。对 $\Delta z_{inv} = z_{inv,next} - z_{inv,t}$ 提取帧级净效应后做向量量化(EMA+死码重启)，输出事件码索引、commitment损失及量化前的效应向量。
* **`GeneratorBank`** (`net/effect_tokenizer.py`)
  * **职责**：可逆连续生成元算子组。对 $z_{rev}$ 通过自适应的线性 basis 映射产生生成元方向，并利用有界预测系数进行线性叠加，以有界增量形式演化 $z_{rev}$。
* **`EventVocabHead`** (`net/heads.py`)
  * **职责**：不可逆事件码分类头。从预测器输出特征中分类预测事件码概率，用于离散通道的监督。
* **`AffordanceHead`** (`net/heads.py`)
  * **职责**：反事实效应幅度头。回归预测无操作/操作之间的反事实效应 $\|e\|$。
* **`SurpriseHead`** (`net/heads.py`)
  * **职责**：多头未来预测集成。利用多头独立预测未来的头间方差作为认知 surprise。
* **`WorldProbeDecoder`** (`net/world_probe.py:L14`)
  * **职责**：世界信念状态检测薄探针。利用极薄的单隐层 MLP 回归解码出音符的客观物理坐标与色彩属性。

### 1.2 行为蒸馏偏置层 (`train/` 目录)
* **`VPTBiasSidecar`** (`train/vpt/distill_vpt.py:L61`)
  * **职责**：可学全局偏置侧翼网络。仅含有 22 维全局常数偏置，无任何前向特征输入，用于在 logits 空间吸收 VPT 数据集的系统性基率/风格偏置，强迫规划头内容路聚焦于状态关联特征。

### 1.3 诊断评估网络层 (`tools/` 目录)
* **`PredOracle`** (`tools/oracle_idm.py:L509`)
  * **职责**：Oracle 逆动力学评估网络。
* **`PoolHead`** (`tools/oracle_idm.py:L273`)
  * **职责**：针对 patch 平均 $\Delta z$ 特征设计的三层 MLP 读出头网络。
* **`GridHead`** (`tools/oracle_idm.py:L287`)
  * **职责**：针对二维网格 patch 特征设计的 $2\mathrm{D}$ 卷积 + 线性映射读出头网络。

### 1.4 内置 OpenAI VPT 策略库网络层 (`net/vpt_lib/` 目录)
* **`MinecraftAgentPolicy`** (`net/vpt_lib/policy.py:L226`)
  * **职责**：VPT 顶层代理策略网络包装类，执行状态初始化与多头动作概率分布解码。
* **`MinecraftPolicy`** (`net/vpt_lib/policy.py:L82`)
  * **职责**：VPT 核心循环网络。融合图像 CNN 特征、Transformer memory 块和前一步动作做 recurrent 决策。
* **`ImgPreprocessing` / `ImgObsProcess`** (`net/vpt_lib/policy.py:L20, L47`)
  * **职责**：VPT 图像通道预处理及缩放调整网络。
* **`InverseActionPolicy`** (`net/vpt_lib/policy.py:L405`)
  * **职责**：VPT 动作反向生成策略网。
* **`ActionHead`** (`net/vpt_lib/action_head.py:L22`)
  * **职责**：VPT 联合动作解码分类头网络。
* **`ImpalaCNN` / `CnnDownStack` / `CnnBasicBlock`** (`net/vpt_lib/impala_cnn.py:L132, L55, L13`)
  * **职责**：VPT 图像特征提取的 Impala ResNet 卷积骨干网络。
* **`MaskedAttention`** (`net/vpt_lib/masked_attention.py:L97`)
  * **职责**：VPT Transformer block 使用的带 memory cache 记忆的多头掩码自注意力层。
* **`MLP`** (`net/vpt_lib/mlp.py:L8`)
  * **职责**：基础的多层感知机结构层。
* **`NormalizeEwma`** (`net/vpt_lib/normalize_ewma.py:L6`)
  * **职责**：指数滑动标准化（EWMA）层。
* **`ScaledMSEHead`** (`net/vpt_lib/scaled_mse_head.py:L11`)
  * **职责**：带自适应比例因子调整的均方误差回归输出头。
* **`FanInInitReLULayer` / `ResidualRecurrentBlocks` / `ResidualRecurrentBlock`** (`net/vpt_lib/util.py:L23, L91, L132`)
  * **职责**：VPT 使用的带权重缩放初始化层、卷积循环残差链及残差块。
* **`AttentionLayerBase` / `PointwiseLayer` / `SplitCallJoin`** (`net/vpt_lib/xf.py:L229, L403, L457`)
  * **职责**：VPT Transformer 专用的自注意力层、逐点 FFN 线性激活块及张量分支整合连接器。

---

## 2. 逻辑上具有“Net 特性”的核心计算与辅助函数

以下函数在代码库中虽没有作为派生自 `nn.Module` 的类，但它们所完成的功能在逻辑上构成了神经网络特征层、数据流动控制、参数重映射及前置/后置投影层的关键功能：

### 2.1 骨干网络加载与特征映射
* **`load_backbone(kind, repo_override)`** (`net/backbone.py:L18`)
  * **特征**：典型的神经网络组装与加载器。负责调用 `transformers.AutoModel` 加载冻结的 DINO 视觉特征主干，提取其参数并封装，向后提供网络特征接口。
* **`TaskTextEncoder._embed(s)`** (`domains/minecraft/task_text.py:L59`)
  * **特征**：加载 MiniLM 文本编码器网络（`AutoModel`），执行句文本的分词（tokenization）与网络前向计算，并做 Mean Pooling 及特征归一化，扮演文本特征网络层。

### 2.2 正弦/位置编码特征函数
* **`sinusoidal_time_encoding(t_vec, d)`** (已迁至 `blocks/encodings.py`，由 `net/world_model.py` 导入)
  * **特征**：神经网络经典的位置编码（PE）层。接收绝对时间戳向量，生成多频段的 `sin` / `cos` 高维连续空间位置向量，用于在 Transformer 动力学推演中对记忆进行时间自定位。

### 2.3 权重与状态流控制
* **`update_ema_teacher(student, teacher, momentum)`** (`utils/losses.py:L7`)
  * **特征**：底层的网络参数动量滑动更新层运算。通过 `pt.lerp_` 对两个模型的权重进行物理替换，是自监督模型（如 BYOL/JEPA）稳定目标的重要手段。
* **`roll_hist(a_hist, t_hist, hv, action, dt_cur)`** (`train/minecraft/_seq.py:L9` / `train/vpt/distill_vpt.py:L148`)
  * **特征**：网络输入端的时序滑动门控。通过直接平移和拼接动作向量和跳帧时间编码，更新动力学 Transformer 的循环动作输入流。

### 2.4 数据特征前处理与归一化
* **`_to_float_img(img)`** (`train/minecraft/_seq.py:L23` / `train/vpt/distill_vpt.py:L143`)
  * **特征**：网络图像输入前处理层。将原始数据中 uint8 的 `[0, 255]` 像素转换为 float，并将 PCIe 数据搬运流量降低 4 倍，是视觉特征提取网络的前置缩放逻辑。

### 2.5 动作特征 Mu-Law 编解码
* **`camera_to_bin(x)`** (`domains/minecraft/vpt_action.py:L33`)
  * **特征**：将连续视角坐标通过非线性 mu-law 压缩映射到 $11\mathrm{D}$ 离散 bin 索引，常用于连续视角到离散编码的目标分类映射以提高回归优化的数值稳定性。
* **`bin_to_camera(idx)`** (`domains/minecraft/vpt_action.py:L45`)
  * **特征**：`camera_to_bin` 的逆映射。在推理解码或评估分析时，将分箱索引投影还原为连续视角坐标。

### 2.6 自适应网络归一化构造
* **`gn(channels)`** (`utils/nn.py:L7`)
  * **特征**：动态层实例化函数。根据输入通道自适应地计算整除组数，并配置创建对应的 `nn.GroupNorm` 归一化网络层。
