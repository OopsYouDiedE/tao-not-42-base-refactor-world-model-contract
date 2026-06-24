# Codebase Analysis: Architecture, Method Call Graphs, and Code Usage

> **范围（2026-06,统一世界基座重设计——从 blocks/ 算子重新实现、不含 vendored 代码 + domains/tools 展平后）**:原 Δz-JEPA 世界模型
> (`net/world_model.py`/`heads.py`/`effect_tokenizer.py`/`net/rssm.py`/`net/dreamer/`)、
> `train/minecraft/*` 训练循环与 `tools/oracle_idm.py` **已删除退役**(见 git 历史)。本文只走读
> **当前仓库实际存在**的代码;新基座 `net/` 实现待构建期落地后增补。`domains/` 已展平进
> `train/<game>/`,`tools/` 已并入 `tests/` 与 `train/minecraft/`。

本文梳理当前代码库各文件、类与方法的功能(第一~四部分),并标识活跃与预留(未接入)代码(第五部分)。

---

## 第一部分：`blocks/` (积木组件库)

#### [blocks/](../blocks/) (各积木组件细分至 encodings.py, dynamics.py, regularization.py, attention.py, conv.py, quantization.py, sequence.py, distributions.py, encoder.py, decoder.py 中)
* **`class ConvGRUCell(nn.Module)`**: 2D 卷积 GRU 单元。通过卷积运算更新门控和候选状态，以 `(1-z)*h + z*n` 的凸组合公式更新隐状态。
* **`class GatedResidual(nn.Module)`**: 带受限增益的门控残差连接。对更新量乘以有界的学习参数 $\gamma$（限制在 $\pm g_{max}$ 内），用于控制递归和残差网络中的数值稳定性。
* **`class FiLM(nn.Module)`**: 特征线性调制模块。通过一个零初始化的双层 MLP 预测乘性系数 $\gamma$ 和加性偏置 $\beta$，对特征图执行 `x * (1 + gamma) + beta` 的条件仿射变换。
* **`class PreLNAttn(nn.Module)`**: 预归一化（Pre-LN）多头注意力模块。支持自注意力（self-attention）或交叉注意力（cross-attention）。当 `store_attn=True` 时进入慢速模式并保存 attention 权重。
* **`class PositionalEmbed(nn.Module)`**: 2D 正弦位置编码模块。产生 `[1, d, H, W]` 的位置编码，在 x 和 y 方向分别使用正弦和余弦频率。
* **`class ProtoDecode(nn.Module)`**: 线性系数与掩码原型合成解码模块。利用 einsum 对系数和原型特征做矩阵乘积，并使用有界 Sigmoid 激活输出局部掩码概率图。
* **`class StochLatent(nn.Module)`**: 随机潜变量采样模块。支持重参数化的高斯采样或直通估计（straight-through estimator）的 Gumbel-Softmax 离散分类分布采样，并计算输出 KL 散度。
* **`class SIGReg(nn.Module)`**: 基于经验高斯分布的 sliced 正则化模块。通过随机投影将高维嵌入降到 1D，利用 Epps-Pulley 经验特征函数检验，配合自适应积分梯形权重，将投影分布对齐到标准正态分布，以防止表征坍缩。
* **`class BoundedActivation(nn.Module)`**: 数值有界激活函数类。执行各类有界激活（深度、光流、位置、概率），对应 exponential clamp、tanh scaling、softplus 等。
* **`class Accumulator(nn.Module)`**: 仿 NALU/NAC 的精确计数累加器。使用限制在 `(-1, 1)` 的伪离散权重，以改善数值范围外推的泛化性。
* **`class DiscreteRouter(nn.Module)`**: 可微的离散分支路由器。训练时使用直通式 Gumbel-Softmax 做硬采样选择，评估时直接 argmax 取独热分支。
* **`class ContinuousTimeEncoding(nn.Module)`**: 连续时间（帧跨度）正弦编码模块。以帧为单位对可变时间 $\Delta t$ 产生正弦和余弦高维编码特征。
* **`class SpatialPosEmbed(nn.Module)`**: 傅里叶特征空间坐标位置编码模块。对注视裁剪（fovea）的 2D 坐标 `(x, y)` 及其尺度对数 `log(s)` 做多频段傅里叶变换并映射成特征嵌入。

---

## 第二部分：`net/` (网络组件)

#### [net/backbone.py](../net/backbone.py)
* **`load_backbone(kind, repo_override=None)`**: 视觉骨干加载函数。通过 HuggingFace 加载冻结的 `dinov3` 或 `dinov2` 模型。返回骨干网络 Module 实例、patch 边长、隐藏状态维度与 register token 数量。

#### [net/config.py](../net/config.py)
* 结构 schema(纯 dataclass,无 IO):模型 d/N/K/J 与各部件选择的类型化配置,配 `build_*` 工厂。`net/` 不读 yaml、不 import 数据层。

#### `net/dreamerv3/` (DreamerV3 世界模型,可训练)
* 从 `blocks/` 算子库重建：`rssm.py`（`RSSM`：GRU 确定性 deter + 离散随机隐变量）、`world_model.py`（`WorldModel`：编码 + RSSM + 解码 + reward/cont 头）、`behavior.py`（`ImagBehavior`：想象 rollout 上的 actor-critic）、`planner.py`（稀疏规划）、`agent.py`（`DreamerV3` + `build_dreamerv3` 工厂）、`config.py`。方法级结构与训练手册见 [dreamer.md](dreamer.md) §1。

#### `net/dreamer4/` (Dreamer4 时空 Transformer 世界模型,仅构建)
* 从 `blocks/` 组装：`tokenizer.py`、`dynamics.py`（`SpaceTimeTransformer` + `ShortcutHead`）、`world_model.py`、`agent.py`（`build_dreamer4` 工厂）、`config.py`。本仓只构建、不提供训练循环。详见 [dreamer.md](dreamer.md) §2。

#### `net/ppo_ad/` (Crafter PPO + Achievement Distillation)
* `actor_critic.py`（`ActorCritic`）、`config.py`（`PPOADConfig`）。供 `train/crafter/train_ppo_ad.py` 使用。

#### `net/vpt_lib/` (vendored OpenAI VPT)
* 原样 vendored 的第三方策略库(见其 `NOTICE`);旧蒸馏 teacher 适配器 `vpt_teacher.py` 已随退役管线删除,新基座蒸馏入口待补;不受本仓代码规范约束。

> 旧 Δz-JEPA 主干 `net/world_model.py`、`heads.py`、`effect_tokenizer.py`、`net/rssm.py` 与对照用 vendored `net/dreamer/` 均已退役删除（见 git 历史）。

---

## 第三部分：`train/` (训练域:数据契约 + 循环,按数据集分目录)

> `domains/` 层已展平进此处:不同数据集的区分全压在 `train/<game>/` 这一层,各自自洽。

### `train/minecraft/` (VPT/BASALT 数据集域)

#### [train/minecraft/vpt_action.py](../train/minecraft/vpt_action.py)
* **`camera_to_bin(x)`**: 连续相机移动值映射为 mu-law 归一化的 11D 离散 bin 索引。
* **`bin_to_camera(idx)`**: 分箱索引逆向还原为连续相机移动值（常数中心点）。
* **`encode_vpt_jsonl(d)`**: 单帧 VPT jsonl 数据转化为 `ACTION_DIM=22` 维契约张量（首两位为鼠标 dx/dy，后二十位为按键状态）。

#### [train/minecraft/vpt_dataset.py](../train/minecraft/vpt_dataset.py)
* **`_action_vec(act_dict, camera_scale)`**: 提取单帧 jsonl 字典动作为归一化动作张量。
* **`_pair_list(data_dir)`**: 寻找目录下所有匹配对的 `.mp4` 视频和 `.jsonl` 动作数据。
* **`_decode_clip(mp4_path, jsonl_path, seq_len)`**: 一次性读取并解码单段 clip 为 tensor 动作和图像的元字典。
* **`class VPTDataset(Dataset)`**: 静态内存预加载的数据集类。
  * **`__getitem__(idx)`**: 随机截取 `seq_len` 的序列窗口，注入随机绝对时间差 `time_offset` 后输出。
* **`class VPTStreamDataset(IterableDataset)`**: 流式滚动加载的数据集类。
  * **`_load_clip(mp4, jsonl)`**: 解码单个 clip 为低分辨率的 uint8 图像数组与动作（128px 下约 0.3 GB/段）。
  * **`_split_actions(act, start, skips)`**: 对切片区间的连续帧动作进行采样合并。输出区间内逐帧原始动作（`act_seq`，右侧零填充）和区间合并动作（`act_agg`：鼠标取区间平均，键盘按过即为 1）。
  * **`__iter__()`**: 流式迭代生成器主循环。在后台线程异步触发 `_spawn_loader` 换入新 clip 段，主循环从内存中的 `clips` 列表随机取片，过滤掉 GUI 占比过大的无效帧，并进行可变跨度 $\Delta t \sim U\{1..frame\_skip\}$ 的 jumpy 采样。

#### [train/minecraft/task_text.py](../train/minecraft/task_text.py)
* **`class TaskTextEncoder`**: 任务描述文本编码器。
  * **`encode(texts)`**: 获取任务文本对应的句向量。内置查表缓存 `_cache` 以避免重复计算。
  * **`_embed(s)`**: 编码逻辑。`"minilm"` 模式下加载 MiniLM 提取句特征并做 $\ell_2$ 归一化；`"mock"` 降级模式下基于字符串 md5 产生确定性的归一化随机向量（用于在无网络或显存不足时区分任务类型）。

> 旧训练循环 `train_minecraft.py`/`losses.py`/`eval.py`/`_seq.py`/`minecraft_viz.py`、`train/vpt/distill_vpt.py` 与蒸馏 teacher 适配器 `vpt_teacher.py`(`VPTTeacher`,minerl-free VPT 边缘化 soft 蒸馏适配器)已随退役管线删除,新基座训练入口待补。

### `train/crafter/` (Crafter 数据集域,当前最活跃)

* **`env.py`**: `VecCrafterEnv`（单进程内顺序步进 n_envs 个 Crafter 实例，Colab 兼容）与 `SubprocVecCrafterEnv`（多 env 分摊到 spawn 子进程并行，高吞吐）。两者 `step()` 返回签名一致：`obs / rew / done / info / new_achievements`，训练循环可零改动互换。
* **`ad_buffer.py`**: `AchievementBuffer`（按成就名缓存示范片段，供 Achievement Distillation）；常量 `ACHIEVEMENTS`（22 个）、`HARD_ACHIEVEMENTS`（科技树硬墙成就）；`covered_names()` 返回已解锁成就。
* **`dreamer_buffer.py`**: `SequenceReplay`（定长序列回放，CPU/uint8 存储 obs，采样小批量再转 GPU；容量 non-wrapping）。
* **`rollout.py`**: `RolloutBuffer`（PPO on-policy 回放）。
* **`ppo_loss.py`**: `ppo_loss`（裁剪式 PPO 策略 + 价值 + 熵损失）。
* **`goal.py` / `state_cache.py`**: 目标条件 actor 与 Go-Explore 式课程相关支撑（配合 `net/dreamerv3/planner.py` 的稀疏规划，默认关闭）。
* **`train_dreamerv3.py`**: DreamerV3 训练循环（采集 ↔ 世界模型更新 ↔ 想象 actor-critic；wm 与 actor/critic 各自独立优化器）。`--size {tiny,small,default}` 选规模。训练手册见 [dreamer.md](dreamer.md)。
* **`train_ppo_ad.py`**: PPO + Achievement Distillation 训练循环；`--vec {serial,subproc}` 选向量环境实现。

### `train/godot_meta_rl/` (Godot 40 环境 RL 子系统)

* **`vec_env.py`**: `GodotVecEnv`（SB3 VecEnv 适配）/ `RolloutProgress`。当前 `train/godot_meta_rl/` 仅保留此对接桥。
* 退役 PPO 执行器（`train_ppo*.py`）与诊断/协议测试（`smoke.py`/`diag_montage.py`/`async_min.py`/`test_*.py`/`cleanup_workspace.py`）已于本次重设计清理中删除。跨平台基础设施在 `utils/godot_rl/`。

---

## 第四部分：`utils/` & `tests/` (辅助、离线脚本与测试)

#### [utils/io.py](../utils/io.py)
* **`load_yaml(path)`**: 配置文件读取函数。读取 YAML 格式配置文件并返回 Python 原生 dict。
* **`get_hf_token(colab_secret_name, set_environ)`**: HuggingFace token 解析函数。按环境变量 -> Colab Secret -> 本地 .env 文件 -> 已登录本地缓存的优先级提取 token，并写回 `os.environ` 完成鉴权，支持 Gated 视觉模型（DINOv3）的后台加载。

> 离线脚本 `download_sample_data.py`(合成 VPT 样本生成器)与 `test_dinov3_hf.py`(DINOv3 骨干冒烟)已于本次清理删除。

#### [tests/unit/](../tests/unit/)
* **`test_sigreg.py`** / **`test_spatial_pos_embed.py`**: CPU 可跑的纯单元测试（SIGReg 防坍缩统计量、空间位置编码形状/连续性/数值安全）。

---

## 第五部分：代码使用状态:活跃 vs 预留

### 1. 活跃 (Active)
* **`net/dreamerv3/` + `train/crafter/`**：Crafter 上可训练的 DreamerV3（世界模型 + 想象 actor-critic + 稀疏 planner）与 PPO+AD，当前最活跃子系统。
* **`blocks/`**：编解码 / RSSM GRU / 分布 / 序列等算子被 `net/dreamerv3`、`net/dreamer4` 复用；其余算子作为统一基座的候选组件来源。
* **`net/backbone.py`**：`load_backbone` 加载冻结视觉骨干；**`net/config.py`**：结构 schema；**`net/vpt_lib/`**：vendored VPT。
* **`train/minecraft/`**：`vpt_action`/`vpt_dataset`/`task_text` 数据契约（训练循环待新基座补）。
* **`train/godot_meta_rl/` + `assets/godot_meta_rl/`**：Godot RL 子系统（见其文档）。
* **`utils/io.py`**：YAML 读取 + HF token；**`utils/godot_rl/`**：Godot 跨平台共享内存基础设施。
* **`tests/`**：`unit/`（SIGReg / 空间位置编码）+ `integration/`（DreamerV3 构建冒烟）。

### 2. 预留:实现存在,当前未接入主循环 (Inactive / Reserved)
* **`net/dreamer4/`**：结构已落地、shape 契约跑通，但无训练循环（流匹配训练待补）。
* **`blocks/` 中的预留算子**：`ConvGRUCell`、`GatedResidual`、`FiLM`、`PositionalEmbed`、`ProtoDecode`、`StochLatent`、`BoundedActivation`、`Accumulator`、`DiscreteRouter`、`SpatialPosEmbed`——保留为积木库,供未来感知/动力学/局部注视 fovea 采样等接入备选。几何/流采样组件(`Warp`/`GlobalTransformApply`/`BEVSplat`/`LocalCorr`/`SoftArgmaxFlow`/`box_iou`/`rot6d_to_matrix`/`make_4x4`)与 `blocks/yolo.py` 已于本次清理删除。
