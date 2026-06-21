# Codebase Analysis: Architecture, Method Call Graphs, and Code Usage

> ⚠️ **范围（2026-06,统一世界基座清白重设计 + domains/tools 展平后）**:原 Δz-JEPA 世界模型
> (`net/world_model.py`/`heads.py`/`effect_tokenizer.py`/`net/rssm.py`/`net/dreamer/`)、
> `train/minecraft/*` 训练循环与 `tools/oracle_idm.py` **已删除退役**(见 git 历史)。本文只走读
> **当前仓库实际存在**的代码;新基座 `net/` 实现待构建期落地后增补。`domains/` 已展平进
> `train/<game>/`,`tools/` 已并入 `tests/` 与 `train/minecraft/`。

本文梳理当前代码库各文件、类与方法的功能(第一~四部分),并标识活跃与预留(未接入)代码(第五部分)。

---

## 第一部分：`blocks/` (积木组件库)

#### [blocks/](file:///c:/Users/zznZZ/Desktop/tao-not-42-base-refactor-world-model-contract/blocks/) (各积木组件已重构并细分至 spatial.py, similarity.py, encodings.py, dynamics.py, regularization.py, attention.py 中)
* **`_base_grid(h, w, device, dtype)`**: 辅助函数。创建大小为 `[2, H, W]` 的 2D 坐标采样网格，坐标顺序为 `(x, y)`。
* **`class Warp(nn.Module)`**: 局部光流重采样模块。输入特征图与光流位移，通过 `grid_sample` 进行双线性凸插值采样，坐标计算使用双精度/fp32以确保数值稳定性。
* **`class GlobalTransformApply(nn.Module)`**: 全局仿射屏幕空间变换模块。使用给定的 `theta` 矩阵对输入特征图进行仿射变换采样。
* **`class LocalCorr(nn.Module)`**: 有界半径的余弦互相关模块。首先对两个输入特征在通道维上做 $\ell_2$ 归一化（$\epsilon=1e-4$），再在限制半径内计算成对的余弦相似度特征图。
* **`class SoftArgmaxFlow(nn.Module)`**: 软相关性流估计模块。通过对相关性矩阵进行 `softmax` 运算，并乘以预存的位置偏置网格，输出期望的 $2\mathrm{D}$ 运动位移。
* **`class ConvGRUCell(nn.Module)`**: 2D卷积 GRU 单元。通过卷积运算更新门限和候选状态，以非扩张的凸更新公式 `(1-z)*h + z*n` 融合记忆。
* **`class GatedResidual(nn.Module)`**: 带受限增益的门控残差连接。对更新量乘以有界的学习参数 $\gamma$（限制在 $\pm g_{max}$ 内），以控制递归和残差深度下的数值稳定性。
* **`class FiLM(nn.Module)`**: 特征线性调制模块。通过一个零初始化的双层 MLP 预测乘性调节系数 $\gamma$ 和加性偏置 $\beta$，以对特征图做 `x * (1 + gamma) + beta` 的条件仿射调制。
* **`class PreLNAttn(nn.Module)`**: 预归一化（Pre-LN）多头注意力模块。支持自注意力（self-attention）或交叉注意力（cross-attention）。当 `store_attn=True` 时进入慢速模式并保存 attention 权重。
* **`class PositionalEmbed(nn.Module)`**: 2D正弦位置编码模块。产生 `[1, d, H, W]` 的位置编码，利用正弦和余弦频率嵌入 x 和 y 方向。
* **`class ProtoDecode(nn.Module)`**: 线性系数与掩码原型合成解码模块。利用 einsum 对系数和原型特征做矩阵乘积，并使用有界 Sigmoid 激活输出局部掩码概率图。
* **`class StochLatent(nn.Module)`**: 随机潜变量采样模块。支持重参数化的高斯采样或直通估计（straight-through estimator）的 Gumbel-Softmax 离散分类分布采样，计算并输出 KL 散度。
* **`class SIGReg(nn.Module)`**: 经验高斯分布 sliced 正则化模块。通过随机投影将高维嵌入降到 $1\mathrm{D}$，利用 Epps-Pulley 经验特征函数检验，配合自适应的积分梯形权重，将投影分布对齐到标准正态分布，防表征坍缩。
* **`rot6d_to_matrix(x, eps)`**: 辅助函数。使用三维 Gram-Schmidt 正交化，将输入的 6D 向量转换成 $SO(3)$ 旋转矩阵。
* **`make_4x4(R, t)`**: 辅助函数。将 $3\times3$ 旋转矩阵 `R` 与 $3\times1$ 平移向量 `t` 拼接成 $4\times4$ 的齐次坐标变换矩阵。
* **`box_iou(a, b, kind, eps)`**: 辅助函数。计算 xyxy 格式边界框集合的 IoU 或 GIoU。
* **`class BoundedActivation(nn.Module)`**: 数值有界激活函数类。执行各类有界激活机制（深度、光流、位置、概率），对应 exponential clamp、tanh scaling、softplus 等。
* **`class Accumulator(nn.Module)`**: 仿 NALU/NAC 的高精度精确计数累加器。使用限制在 `(-1, 1)` 的伪离散权重，以确保数值范围外推的泛化性能。
* **`class DiscreteRouter(nn.Module)`**: 可微的离散分支路由器。使用直通式 Gumbel-Softmax 在训练时做硬采样选择，在评估时直接 argmax 取独热分支。
* **`class BEVSplat(nn.Module)`**: 3D-to-BEV 投影 splatting 模块。结合相机内参和外参（位姿），将像素特征及对应深度投影到 $3\mathrm{D}$ 世界坐标系，并 scatter量化累加到俯视 BEV 栅格中。
* **`class ContinuousTimeEncoding(nn.Module)`**: 连续时间（帧跨度）正弦编码模块。以帧为单位对可变时间 $\Delta t$ 产生可微的正弦和余弦高维编码特征。
* **`class SpatialPosEmbed(nn.Module)`**: 傅里叶特征空间坐标位置编码模块。对注视裁剪（fovea）的 $2\mathrm{D}$ 坐标 `(x, y)` 及其尺度对数 `log(s)` 做多频段傅里叶变换并映射成特征嵌入。

#### [blocks/yolo.py](file:///c:/Users/zznZZ/Desktop/tao-not-42-base-refactor-world-model-contract/blocks/yolo.py)
* **`autopad(k, p, d)`**: 辅助函数。自动计算填充以使卷积后的空间分辨率在步长为 1 时保持不变。
* **`class Conv(nn.Module)`**: 标准的 `Conv2d - BatchNorm2d - SiLU` 组合模块。
* **`class DWConv(Conv)`**: 深度可分离卷积模块。
* **`class Bottleneck(nn.Module)`**: 残差瓶颈模块。
* **`class C3(nn.Module)`**: CSP Bottleneck 结构模块，使用 3 个卷积和 Bottleneck 链。
* **`class C3k(C3)`**: 允许自定义卷积核大小的 C3 变体模块。
* **`class C2f(nn.Module)`**: YOLOv8 中的 CSP 多分支跨级特征融合模块。
* **`class C3k2(C2f)`**: YOLO11 变体模块。在 C2f 结构中支持 `PSABlock` 注意力或 `C3k` 的开关配置。
* **`class Attention(nn.Module)`**: 仿 YOLOE 的多头空间通道自注意力模块。
* **`class PSABlock(nn.Module)`**: 位置敏感注意力（Position-Sensitive Attention）块，由 `Attention` 和两层线性 FFN 串联而成。
* **`class C2PSA(nn.Module)`**: 融合 CSP 和 `PSABlock` 的高阶特征模块。
* **`class SPPF(nn.Module)`**: 快速空间金字塔池化模块。通过并行串联的 `MaxPool2d` 提取多尺度空间特征。
* **`class Concat(nn.Module)`**: 沿通道维或指定维的张量拼接层。

---

## 第二部分：`net/` (网络组件)

#### [net/backbone.py](file:///c:/Users/zznZZ/Desktop/tao-not-42-base-refactor-world-model-contract/net/backbone.py)
* **`load_backbone(kind, repo_override=None)`**: 视觉骨干加载函数。通过 HuggingFace 加载冻结的 `dinov3` 或 `dinov2` 模型。返回骨干网络 Module 实例、patch 边长、隐藏状态维度与 register token 数量。

#### [net/config.py](file:///c:/Users/zznZZ/Desktop/tao-not-42-base-refactor-world-model-contract/net/config.py)
* 结构 schema(纯 dataclass,无 IO):模型 d/N/K/J 与各部件选择的类型化配置,配 `build_*` 工厂。`net/` 不读 yaml、不 import 数据层。

#### `net/vpt_lib/` (vendored OpenAI VPT)
* 原样 vendored 的第三方策略库(见其 `NOTICE`),被 `train/minecraft/vpt_teacher.py` 用作蒸馏 teacher;不受本仓代码规范约束。

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

#### [train/minecraft/vpt_teacher.py](file:///c:/Users/zznZZ/Desktop/tao-not-42-base-refactor-world-model-contract/train/minecraft/vpt_teacher.py) (原 `tools/vpt_teacher.py`)
* **`class VPTTeacher`**: `minerl-free` 版本的 OpenAI VPT 代理包装器（加载 `.model`+`.weights`，依赖 vendored `net/vpt_lib`）。
  * **`step(img_rgb)`**: 接收图像前向并更新其 Transformer 内部隐状态。
  * **`to_contract(pd)`**: 将联合动作分布边缘化降维并重排，映射成我们 22 维契约的 soft 动作目标（用于蒸馏）。

> 旧训练循环 `train_minecraft.py`/`losses.py`/`eval.py`/`_seq.py`/`minecraft_viz.py` 与 `train/vpt/distill_vpt.py` 已随退役管线删除,新基座训练入口待补。

### `train/godot_meta_rl/` (Godot 40 环境 RL 子系统,活跃)

* **`vec_env.py`**: `GodotVecEnv`（SB3 VecEnv 适配）/ `RolloutProgress`。
* **`train_ppo*.py`**: 锁步 / 线程异步 / 双进程 三种 PPO 执行器。
* **`smoke.py` / `diag_montage.py` / `async_min.py` / `test_*.py`**: 冒烟·诊断·共享内存协议测试（需活 Godot 进程，紧贴环境，不进 `tests/`）。跨平台基础设施在 `utils/godot_rl/`。

---

## 第四部分：`utils/` & `tests/` (辅助、离线脚本与测试)

#### [utils/io.py](file:///c:/Users/zznZZ/Desktop/tao-not-42-base-refactor-world-model-contract/utils/io.py)
* **`load_yaml(path)`**: 配置文件读取函数。读取 YAML 格式预设配置文件并返回 Python 原生 plain dict 字典。
* **`get_hf_token(colab_secret_name, set_environ)`**: HuggingFace token 双重解析助手。按环境变量 -> Colab Secret -> 本地 .env 文件 -> 已登录本地缓存的优先级提取 Token，并写回 `os.environ` 自动完成鉴权，支持 Gated 视觉模型（DINOv3）的后台加载。

#### [tests/download_sample_data.py](file:///c:/Users/zznZZ/Desktop/tao-not-42-base-refactor-world-model-contract/tests/download_sample_data.py) (原 `tools/`)
* **`main()`**: 离线合成 VPT 样本生成器（动作与画面强相关 + 不可逆状态 + `--counterfactual` 反事实分支族）。产出 `VPTStreamDataset` 兼容的成对 `.mp4`+`.jsonl`，供离线管线冒烟。

#### [tests/test_dinov3_hf.py](file:///c:/Users/zznZZ/Desktop/tao-not-42-base-refactor-world-model-contract/tests/test_dinov3_hf.py) (原 `tools/`)
* **`main()`**: 命令行冒烟入口。从 HuggingFace 缓存加载 DINOv3 (ViT-B/16)，在 128×128 尺度下执行前向运算以核对 patch/register token 形状、归一化常数与 VRAM/时间开销。

#### [tests/unit/](file:///c:/Users/zznZZ/Desktop/tao-not-42-base-refactor-world-model-contract/tests/unit/)
* **`test_sigreg.py`** / **`test_spatial_pos_embed.py`**: CPU 可跑的纯单元测试（SIGReg 防坍缩统计量、空间位置编码形状/连续性/数值安全）。

---

## 第五部分：代码使用状态:活跃 vs 预留

### 1. 活跃 (Active)
* **`blocks/`**：`ContinuousTimeEncoding`、`PreLNAttn`、`SIGReg` 等算子,作为新基座"请客"来源。
* **`net/backbone.py`**：`load_backbone` 加载冻结视觉骨干；**`net/config.py`**：结构 schema；**`net/vpt_lib/`**：vendored VPT。
* **`train/minecraft/`**：`vpt_action`/`vpt_dataset`/`task_text` 数据契约 + `vpt_teacher` 蒸馏 teacher。
* **`train/godot_meta_rl/`**：Godot RL 训练/诊断/协议测试（活跃子系统）。
* **`utils/io.py`**：YAML 读取 + HF token；**`utils/godot_rl/`**：Godot 跨平台共享内存基础设施。
* **`tests/`**：`unit/`（CPU 单测）+ 离线脚本（`download_sample_data` 合成数据 / `test_dinov3_hf` 骨干冒烟）。

### 2. 预留:实现存在,当前未接入主循环 (Inactive / Reserved)
* **`blocks/yolo.py`**：冻结 DINO 后弃用从头训练的 CNN/YOLO 局部感知骨干，`Conv`/`C3`/`C2f`/`SPPF` 等全闲置，按重构评审保留。
* **`blocks/` 中的几何/流采样组件**：`Warp`、`GlobalTransformApply`、`LocalCorr`、`SoftArgmaxFlow`、`ConvGRUCell`、`GatedResidual`、`FiLM`、`PositionalEmbed`、`ProtoDecode`、`StochLatent`、`rot6d_to_matrix`、`make_4x4`、`box_iou`、`BoundedActivation`、`Accumulator`、`DiscreteRouter`、`BEVSplat`、`SpatialPosEmbed`——保留为积木库,供未来三维几何重建或局部注视 fovea 采样等备选方案平替。
