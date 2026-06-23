# Dreamer 系世界模型 — 设计与目录说明

> 本文档说明 **Dreamer 系**(DreamerV3 + Dreamer4)的整体设计与目录结构。SSOT：数值逻辑以
> `net/dreamerv3/`、`net/dreamer4/`、`blocks/` 代码为准，本文只解释架构意图与目录组织。

## 0. 现状(2026-06，从头重建)

仓库早先 vendored 过一份完整的 DreamerV3（NM512/dreamerv3-torch，MIT），后整目录退役删除，
**仅保留被 `blocks/` 复用的独立算子**：`GRUCell`（blocks/dynamics.py）、`Conv2dSamePad`/
`ImgChLayerNorm`（blocks/conv.py）、`ConvEncoder`/`ConvDecoder`（blocks/encoder.py、decoder.py）、
symlog·two-hot·OneHot 等分布（blocks/distributions.py）、`static_scan`·`lambda_return`
（blocks/sequence.py）。MIT 署名/许可证见 `blocks/NOTICE.dreamerv3`、`blocks/LICENSE.dreamerv3`。

当前的 `net/dreamerv3/` 与 `net/dreamer4/` 是**从 `blocks/` 算子库重新实现**的
（**非 vendored、非逐字照抄**）：L2 网络层把与任务无关的算子组装成两个完整的世界模型。
因此两者受本仓代码规范（I1–I8、写作纪律、依赖方向 blocks←net←train）约束，与旧 vendored 目录不同。

## 1. DreamerV3（`net/dreamerv3/`）— 可训练基线

对应 Hafner 等《Mastering Diverse Domains through World Models》（arXiv:2301.04104）的结构。
训练任务：Crafter（64×64 RGB，Discrete(17)）。

| 文件 | 内容 |
|---|---|
| `config.py` | `DreamerV3Config` 纯 dataclass（RSSM/编解码/头/想象的结构超参，默认对齐上游 defaults） |
| `rssm.py` | `RSSM`：`GRUCell` 确定性 deter + 离散 32×32 随机隐变量（`OneHotDist` unimix）；`observe`/`imagine_with_action`（`static_scan` 展开）/`kl_loss`（动力学/表征 KL + free-bits） |
| `world_model.py` | `WorldModel`：`ConvEncoder` + `RSSM` + `ConvDecoder`（`MSEDist` 重建，图像在 [-0.5, 0.5]）+ reward（`DiscDist` two-hot symexp）+ cont（`Bernoulli`）头；`loss()` 计算重建+奖励+终止+KL |
| `behavior.py` | `ImagBehavior`：想象 rollout 上的离散策略（`OneHotDist`）+ two-hot 价值（`DiscDist`）+ 慢靶 critic；`lambda_return` 计算价值目标，`RewardEMA`（5/95 分位）归一化优势，reinforce 策略梯度。rollout 与回报目标全程 `no_grad`（reinforce 不需路径梯度，可节省显存且不改变学习信号） |
| `agent.py` | `DreamerV3`（持有 wm+behavior，`policy()` 递归单步交互）+ `build_dreamerv3(**overrides)` 工厂 |

依赖方向遵约：`net/dreamerv3/` 只 import `blocks` 与本包，**不含训练循环/优化器/数据加载**。
训练循环在 `train/crafter/`：`dreamer_buffer.py`（`SequenceReplay` 定长序列回放，CPU/uint8 存储）+
`train_dreamerv3.py`（采集 ↔ 世界模型更新 + 想象 actor-critic 更新；wm 与 actor/critic 各自有独立优化器，
想象损失不回传梯度到世界模型）。`--size {tiny,small,default}` 选结构规模。

**验证（L4 GPU，tiny，6k 步冒烟）**：世界模型损失单调下降（image 429→179、reward 4.1→0.5、
wm_total 434→181），KL 稳定约 1.6，无 NaN；采集→wm 训练→想象→actor-critic→递归 policy 全链路跑通。
该规模和步数下策略尚未充分收敛（actor lr 3e-5、仅数百次更新），世界模型损失下降是当前可见的正向信号。
CPU 小尺寸前向+反向冒烟见 `tests/integration/test_dreamer_build.py`。

## 2. Dreamer4（`net/dreamer4/`）— 仅构建，暂不训练

对应 Hafner 等《Dreamer 4: Training Agents Inside of Scalable World Models》（2025）的结构骨架，
从 `blocks/` 组装。**本仓只构建、不提供训练循环**（流匹配世界模型训练 + 想象 actor-critic 待补）。

| 文件 | 内容 |
|---|---|
| `config.py` | `Dreamer4Config`（tokenizer/动力学/shortcut 头/各头的结构超参；为本机可构造做了缩放，非论文原尺度） |
| `tokenizer.py` | `Tokenizer`：`ConvEncoder`（flatten=False 取空间特征图）→ 每空间位置一个连续潜 token；可选 `VectorQuantizer` 离散码本瓶颈；`ConvDecoder` 还原图像 |
| `dynamics.py` | `SpaceTimeTransformer`：每块 = 帧内空间自注意（`MHABlock` 非因果）+ 跨帧因果时间自注意（`MHABlock` causal）+ 动作 AdaLN 调制（零初始恒等）；`ShortcutHead`：shortcut-forcing 流匹配速度头（给定上下文+噪声 token+流时间 τ+步长 d → 速度 v，支持少步生成） |
| `world_model.py` | `WorldModel`：tokenizer + 动力学 + shortcut 头 + reward/cont 头；`forward()` 跑一次形状自洽前向（编码→上下文→少步 Euler 流生成→解码） |
| `agent.py` | `Dreamer4`（wm + actor/critic 头）+ `build_dreamer4(**overrides)` 工厂 |

暂不训练的原因：Dreamer4 的训练目标（shortcut forcing 的流匹配 + 一致性、想象 actor-critic）
与 DreamerV3 差异较大，需要单独的训练域实现。当前阶段先按 blocks 分层把结构落地、跑通 shape 契约
（`tests/integration/test_dreamer_build.py` 覆盖连续/VQ 两种 tokenizer 的前向），训练循环后续按需补充。

## 2.5 训练实践笔记（Crafter + L4）

以下为可复用的操作性经验，非训练流水账：

- **吞吐受环境步进主导，GPU 呈突发利用**：Crafter 在 CPU 上顺序步进 n_envs 个实例（纯 Python
  世界生成，`VecCrafterEnv` 为 Colab 兼容不用多进程），env 步进是 wall-clock 瓶颈；单次世界模型
  前向+反向（small=6.7M 参数、batch 16×64）耗时较短，因此 GPU 利用率是**突发性**的——
  单次 `nvidia-smi` 快照若落在两次突发之间的 env 步进间隙，读数会偏低甚至为 0，需多次采样取均值
  （实测 small 配置约 20–80% 突发、显存约 5.8/23 GB）。
- **提升 GPU 利用率和样本效率的主要旋钮是 train ratio**：即 `updates_per`（每次环境迭代的梯度步数），
  其次是 batch/seq/模型规模。本仓 train ratio = `updates_per × batch × seq /(train_every × n_envs)`。
  DreamerV3 原版经典 train ratio ≈ 512（回放帧 / 环境步）；`updates_per=4, train_every=1, batch=16,
  seq=64, n_envs=8` ⇒ 512，正好对齐。较大的 train ratio 可提升样本效率与 GPU 占用，但会增加 wall-clock。
- **回放放 CPU、采样转 GPU**：`SequenceReplay` 以 uint8 在 CPU 存储 obs（节省显存），采样的小批量
  才转到设备；容量 non-wrapping，按 `total_steps/n_envs` 预留，用尽即停止写入（避免环形窗口跨写指针）。
- **想象 rollout 全程 `no_grad`**：reinforce 策略梯度不需沿 rollout 的路径梯度（actor 仅由显式
  logπ·advantage 接收梯度，critic 在 detach 特征上回归），切断 rollout 图可节省显存且不改变学习信号。
- **RSSM 因果对齐**：训练 `observe` 必须喂右移一位的 `prev_action`（进入 obs[t] 的动作 = action[t-1]），
  与想象 `img_step(state, departing_action)` 的因果方向一致；否则训练用"离开当前帧的动作"预测当前状态，
  与想象不一致，会污染想象 rollout 动力学，影响策略学习（见 world_model.py 注释与 git fix）。
- **日志缓冲**：`nohup … > log` 重定向时 Python stdout 默认全缓冲，进度日志长时间不落盘会让进程
  看似卡死；`train_dreamerv3` 已在 `main()` 开头设 stdout 行缓冲（等价 `python -u`）。

### 吞吐瓶颈与优化（L4，small 配置实测）

逐项计时定位瓶颈（单次世界模型更新 = 编码 + RSSM observe + 解码 + reward/cont + KL + 反向）：

| 部件 | 耗时（B=16，T=64） | 性质 |
|---|---|---|
| Crafter env 步进（8 env/iter） | 19 ms | **非瓶颈**（416 env-步/s） |
| encoder | 18 ms | flop-bound，随 batch 线性 |
| **RSSM observe（T=64）** | **363 ms** | **沿 T 逐步、overhead-bound**（与 batch 几乎无关：B=16 与 B=96 同为约 370 ms） |
| decoder image_dist | 61 ms | flop-bound，随 batch×T 线性 |
| reward+cont dist | 1 ms | — |
| behavior.loss（H=15） | 109 ms | 随 B×T 线性 |

关键结论：**瓶颈是 RSSM 沿时间维的逐步 Python 循环（observe），不是 GPU flop，也不是环境**；
batch=16 时 GPU 处于"饥饿"态（每步张量较小，固定的 per-step 开销占主导）。三项优化（均落 `_enable_fast_math()`
与默认配置，不改动 `net/`）：

1. **关闭 `torch.distributions` 参数校验**：observe 每步构造 `Independent(OneHotDist)` 时的 simplex/有限性
   校验是纯 CPU 开销 ⇒ observe **363→166 ms（2.2×）**，单行全局开关，数值不变（只跳过输入校验）。
2. **TF32 + cudnn.benchmark**：加速 flop-bound 的编/解码器（固定形状）。
3. **大 batch + 短 seq**：observe overhead-bound ⇒ 加大 batch 近乎免费地多喂数据，缩短 seq 线性减少耗时。

三者叠加把单更新吞吐从 **1304 → 4412 帧/s（约 3.4×）**（B=16，T=64 → B=96，T=32）。综合（env 可忽略、
按 train ratio 折算）wall-clock 约 **3–6×** 提速。默认配置已切到 `batch=48, seq=32, updates_per=2,
train_every=1`（ratio=384）。显存：B=64，T=48 峰值约 20 GB，B≤64，T≤32 安全（<14 GB）；
seq/batch×T 是显存与解码/想象成本的主因。进一步可选 torch.compile 编/解码器（未做，收益低于上述优化）。

## 3. 升级/借鉴关系

- DreamerV3 的随机隐变量（离散 32×32）是本仓统一世界基座里 ξ 思想的来源（见 mental_world）。
- 两个模型共享 `blocks/` 的同一批算子（GRU/卷积/分布/序列/注意力/MLP），互为对照：
  DreamerV3 用 GRU 递归 + 离散 RSSM，Dreamer4 用时空 Transformer + 连续 token + 流匹配生成。
