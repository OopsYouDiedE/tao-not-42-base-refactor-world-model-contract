# Codebase Analysis: Architecture, Method Call Graphs, and Code Usage

> ⚠️ **范围（2026-06,统一世界基座清白重设计 + domains/tools 展平后）**:原 Δz-JEPA 世界模型
> (`net/world_model.py`/`heads.py`/`effect_tokenizer.py`/`net/rssm.py`/`net/dreamer/`)、
> `train/minecraft/*` 训练循环与 `tools/oracle_idm.py` **已删除退役**(见 git 历史)。本文只走读
> **当前仓库实际存在**的代码;新基座 `net/` 实现待构建期落地后增补。`domains/` 已展平进
> `train/<game>/`,`tools/` 已并入 `tests/` 与 `train/minecraft/`。

本文梳理当前代码库各文件、类与方法的功能(第一~四部分),并标识活跃与预留(未接入)代码(第五部分)。

---

## 第一部分：`blocks/` (积木组件库)

#### [blocks/](file:///c:/Users/zznZZ/Desktop/tao-not-42-base-refactor-world-model-contract/blocks/) (各积木组件细分至 encodings.py, dynamics.py, regularization.py, attention.py, conv.py, quantization.py, sequence.py, distributions.py, encoder.py, decoder.py 中)
* **`class ConvGRUCell(nn.Module)`**: 2D卷积 GRU 单元。通过卷积运算更新门限和候选状态，以非扩张的凸更新公式 `(1-z)*h + z*n` 融合记忆。
* **`class GatedResidual(nn.Module)`**: 带受限增益的门控残差连接。对更新量乘以有界的学习参数 $\gamma$（限制在 $\pm g_{max}$ 内），以控制递归和残差深度下的数值稳定性。
* **`class FiLM(nn.Module)`**: 特征线性调制模块。通过一个零初始化的双层 MLP 预测乘性调节系数 $\gamma$ 和加性偏置 $\beta$，以对特征图做 `x * (1 + gamma) + beta` 的条件仿射调制。
* **`class PreLNAttn(nn.Module)`**: 预归一化（Pre-LN）多头注意力模块。支持自注意力（self-attention）或交叉注意力（cross-attention）。当 `store_attn=True` 时进入慢速模式并保存 attention 权重。
* **`class PositionalEmbed(nn.Module)`**: 2D正弦位置编码模块。产生 `[1, d, H, W]` 的位置编码，利用正弦和余弦频率嵌入 x 和 y 方向。
* **`class ProtoDecode(nn.Module)`**: 线性系数与掩码原型合成解码模块。利用 einsum 对系数和原型特征做矩阵乘积，并使用有界 Sigmoid 激活输出局部掩码概率图。
* **`class StochLatent(nn.Module)`**: 随机潜变量采样模块。支持重参数化的高斯采样或直通估计（straight-through estimator）的 Gumbel-Softmax 离散分类分布采样，计算并输出 KL 散度。
* **`class SIGReg(nn.Module)`**: 经验高斯分布 sliced 正则化模块。通过随机投影将高维嵌入降到 $1\mathrm{D}$，利用 Epps-Pulley 经验特征函数检验，配合自适应的积分梯形权重，将投影分布对齐到标准正态分布，防表征坍缩。
* **`class BoundedActivation(nn.Module)`**: 数值有界激活函数类。执行各类有界激活机制（深度、光流、位置、概率），对应 exponential clamp、tanh scaling、softplus 等。
* **`class Accumulator(nn.Module)`**: 仿 NALU/NAC 的高精度精确计数累加器。使用限制在 `(-1, 1)` 的伪离散权重，以确保数值范围外推的泛化性能。
* **`class DiscreteRouter(nn.Module)`**: 可微的离散分支路由器。使用直通式 Gumbel-Softmax 在训练时做硬采样选择，在评估时直接 argmax 取独热分支。
* **`class ContinuousTimeEncoding(nn.Module)`**: 连续时间（帧跨度）正弦编码模块。以帧为单位对可变时间 $\Delta t$ 产生可微的正弦和余弦高维编码特征。
* **`class SpatialPosEmbed(nn.Module)`**: 傅里叶特征空间坐标位置编码模块。对注视裁剪（fovea）的 $2\mathrm{D}$ 坐标 `(x, y)` 及其尺度对数 `log(s)` 做多频段傅里叶变换并映射成特征嵌入。

---

## 第二部分：`net/` (网络组件)

#### [net/backbone.py](file:///c:/Users/zznZZ/Desktop/tao-not-42-base-refactor-world-model-contract/net/backbone.py)
* **`load_backbone(kind, repo_override=None)`**: 视觉骨干加载函数。通过 HuggingFace 加载冻结的 `dinov3` 或 `dinov2` 模型。返回骨干网络 Module 实例、patch 边长、隐藏状态维度与 register token 数量。

#### [net/config.py](file:///c:/Users/zznZZ/Desktop/tao-not-42-base-refactor-world-model-contract/net/config.py)
* 结构 schema(纯 dataclass,无 IO):模型 d/N/K/J 与各部件选择的类型化配置,配 `build_*` 工厂。`net/` 不读 yaml、不 import 数据层。

#### `net/vpt_lib/` (vendored OpenAI VPT)
* 原样 vendored 的第三方策略库(见其 `NOTICE`);旧蒸馏 teacher 适配器 `vpt_teacher.py` 已随退役管线删除,新基座蒸馏入口待补;不受本仓代码规范约束。

> 旧 Δz-JEPA 主干 `net/world_model.py`、`heads.py`、`effect_tokenizer.py` 与对照用 `net/dreamer/` 均已退役删除,新基座 `net/` 实现待构建期补入。

---

## 第三部分：`train/` (训练域:数据契约 + 循环,按数据集分目录)

> `domains/` 层已展平进此处:不同数据集的区分全压在 `train/<game>/` 这一层,各自自洽。

### `train/minecraft/` (VPT/BASALT 数据集域)

#### [train/minecraft/vpt_action.py](file:///c:/Users/zznZZ/Desktop/tao-not-42-base-refactor-world-model-contract/train/minecraft/vpt_action.py)
* **`camera_to_bin(x)`**: 连续相机移动值映射为 mu-law 归一化的 $11\mathrm{D}$ 离散 bin 索引。
* **`bin_to_camera(idx)`**: 分箱索引逆向还原为连续相机移动值（常数中心点）。
* **`encode_vpt_jsonl(d)`**: 单帧 VPT jsonl 数据转化为 `ACTION_DIM=22` 维契约张量（首两位为鼠标 dx/dy，后二十位为按键状态）。

#### [train/minecraft/vpt_dataset.py](file:///c:/Users/zznZZ/Desktop/tao-not-42-base-refactor-world-model-contract/train/minecraft/vpt_dataset.py)
* **`_action_vec(act_dict, camera_scale)`**: 提取单帧 jsonl 字典动作为归一化动作张量。
* **`_pair_list(data_dir)`**: 寻找目录下所有匹配对的 `.mp4` 视频和 `.jsonl` 动作数据。
* **`_decode_clip(mp4_path, jsonl_path, seq_len)`**: 一次性读取并解码单段 clip 为 tensor 动作和图像的元字典。
* **`class VPTDataset(Dataset)`**: 静态内存预加载的数据集类。
  * **`__getitem__(idx)`**: 随机截取 `seq_len` 的序列窗口，注入随机绝对时间差 `time_offset` 后输出。
* **`class VPTStreamDataset(IterableDataset)`**: 流式滚动加载的高性能数据集类。
  * **`_load_clip(mp4, jsonl)`**: 解码单个 clip 为低分辨率的 uint8 图像数组与动作，以此在常驻内存中只占极小体积（$128\mathrm{px}$ 下仅 $\approx0.3\mathrm{GB}$/段）。
  * **`_split_actions(act, start, skips)`**: 对切片区间的连续帧动作进行采样合并。输出区间内逐帧原始动作（`act_seq`，右侧零填充）和区间合并动作效应（`act_agg`：鼠标取区间平均，键盘按过即为 1）。
  * **`__iter__()`**: 流式迭代生成器主循环。在后台线程异步触发 `_spawn_loader` 换入全新 clip 段，在主前向循环中直接从内存中的 `clips` 列表随机取片，过滤掉 GUI 占比过大的无效帧，并进行可变跨度 $\Delta t \sim U\{1..frame\_skip\}$ 的 jumpy 采样。

#### [train/minecraft/task_text.py](file:///c:/Users/zznZZ/Desktop/tao-not-42-base-refactor-world-model-contract/train/minecraft/task_text.py)
* **`class TaskTextEncoder`**: 任务描述文本编码器。
  * **`encode(texts)`**: 获取任务文本对应的句向量。内置查表缓存机制 `_cache` 以消除重复计算。
  * **`_embed(s)`**: 编码逻辑。在 `"minilm"` 模式下加载 MiniLM 提取句特征并做 $\ell_2$ 归一化；在 `"mock"` 降级模式下基于字符串 md5 产生确定性的归一化随机向量（确保在没有网络/显存不足时能区分任务类型）。

> 旧训练循环 `train_minecraft.py`/`losses.py`/`eval.py`/`_seq.py`/`minecraft_viz.py`、`train/vpt/distill_vpt.py` 与蒸馏 teacher 适配器 `vpt_teacher.py`(`VPTTeacher`,minerl-free VPT 边缘化 soft 蒸馏适配器)已随退役管线删除,新基座训练入口待补。

### `train/godot_meta_rl/` (Godot 40 环境 RL 子系统,活跃)

* **`vec_env.py`**: `GodotVecEnv`（SB3 VecEnv 适配）/ `RolloutProgress`。当前 `train/godot_meta_rl/` 仅保留此对接桥。
* 退役 PPO 执行器（`train_ppo*.py`）与诊断/协议测试（`smoke.py`/`diag_montage.py`/`async_min.py`/`test_*.py`/`cleanup_workspace.py`）已于清白重设计清理中删除。跨平台基础设施在 `utils/godot_rl/`。

---

## 第四部分：`utils/` & `tests/` (辅助、离线脚本与测试)

#### [utils/io.py](file:///c:/Users/zznZZ/Desktop/tao-not-42-base-refactor-world-model-contract/utils/io.py)
* **`load_yaml(path)`**: 配置文件读取函数。读取 YAML 格式预设配置文件并返回 Python 原生 plain dict 字典。
* **`get_hf_token(colab_secret_name, set_environ)`**: HuggingFace token 双重解析助手。按环境变量 -> Colab Secret -> 本地 .env 文件 -> 已登录本地缓存的优先级提取 Token，并写回 `os.environ` 自动完成鉴权，支持 Gated 视觉模型（DINOv3）的后台加载。

> 离线脚本 `download_sample_data.py`(合成 VPT 样本生成器)与 `test_dinov3_hf.py`(DINOv3 骨干冒烟)已于本次清理删除。

#### [tests/unit/](file:///c:/Users/zznZZ/Desktop/tao-not-42-base-refactor-world-model-contract/tests/unit/)
* **`test_sigreg.py`** / **`test_spatial_pos_embed.py`**: CPU 可跑的纯单元测试（SIGReg 防坍缩统计量、空间位置编码形状/连续性/数值安全）。

---

## 第五部分：代码使用状态:活跃 vs 预留

### 1. 活跃 (Active)
* **`blocks/`**：`ContinuousTimeEncoding`、`PreLNAttn`、`SIGReg` 等算子,作为新基座"请客"来源。
* **`net/backbone.py`**：`load_backbone` 加载冻结视觉骨干；**`net/config.py`**：结构 schema；**`net/vpt_lib/`**：vendored VPT。
* **`train/minecraft/`**：`vpt_action`/`vpt_dataset`/`task_text` 数据契约（`vpt_teacher` 蒸馏 teacher 已退役删除，待重建）。
* **`train/godot_meta_rl/`**：Godot RL 训练/诊断/协议测试（活跃子系统）。
* **`utils/io.py`**：YAML 读取 + HF token；**`utils/godot_rl/`**：Godot 跨平台共享内存基础设施。
* **`tests/`**：`unit/`（CPU 单测：SIGReg 防坍缩 / 空间位置编码）。

### 2. 预留:实现存在,当前未接入主循环 (Inactive / Reserved)
* **`blocks/` 中的预留算子**：`ConvGRUCell`、`GatedResidual`、`FiLM`、`PositionalEmbed`、`ProtoDecode`、`StochLatent`、`BoundedActivation`、`Accumulator`、`DiscreteRouter`、`SpatialPosEmbed`——保留为积木库,供未来感知/动力学/局部注视 fovea 采样等接入备选。几何/流采样组件(`Warp`/`GlobalTransformApply`/`BEVSplat`/`LocalCorr`/`SoftArgmaxFlow`/`box_iou`/`rot6d_to_matrix`/`make_4x4`)与 `blocks/yolo.py` 已于本次清理删除。
