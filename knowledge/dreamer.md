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

## 2. Dreamer4（`net/dreamer4/`）— 世界模型可训练（离线 VPT + 在线 CraftGround）

对应 Hafner 等《Dreamer 4: Training Agents Inside of Scalable World Models》（2025）的结构骨架，
从 `blocks/` 组装。世界模型训练已接线（`WorldModel.loss()` + 两个训练域入口）；
想象 actor-critic / 策略蒸馏阶段仍待补。

| 文件 | 内容 |
|---|---|
| `config.py` | `Dreamer4Config`（tokenizer/动力学/shortcut 头/各头的结构超参；为本机可构造做了缩放，非论文原尺度） |
| `tokenizer.py` | `Tokenizer`：`ConvEncoder`（flatten=False 取空间特征图）→ 每空间位置一个连续潜 token；可选 `VectorQuantizer` 离散码本瓶颈；`ConvDecoder` 还原图像 |
| `dynamics.py` | `SpaceTimeTransformer`：每块 = 帧内空间自注意（`MHABlock` 非因果）+ 跨帧因果时间自注意（`MHABlock` causal）+ 动作 AdaLN 调制（零初始恒等）；`ShortcutHead`：shortcut-forcing 流匹配速度头（给定上下文+噪声 token+流时间 τ+步长 d → 速度 v，支持少步生成） |
| `world_model.py` | `WorldModel`：tokenizer + 动力学 + shortcut 头 + reward/cont 头；`forward()` 跑一次形状自洽前向（编码→上下文→少步 Euler 流生成→解码）；`loss()` 世界模型训练损失；`eval_next_frame()` 生成质量评估 |
| `agent.py` | `Dreamer4`（wm + actor/critic 头）+ `build_dreamer4(**overrides)` 工厂 |

### 世界模型训练（2026-07 接线）

`WorldModel.loss()`（置层沿 `net/dreamerv3.WorldModel.loss` 先例）为单循环简化的 Dreamer4 目标：

- **recon**：tokenizer 重建 MSE。tokenizer 只由此项训练——动力学侧对 token `detach()`，
  近似论文的两阶段（先 tokenizer 后动力学）训练。
- **flow**：基础流匹配。x_τ=(1-τ)ε+τz₁、速度目标 v*=z₁-ε，在最小步长 d_min 处监督。
- **sc（shortcut 自一致）**：随机较大步长 d∈{2·d_min…1}（τ 取 d 的 Euler 网格整数倍），
  目标 = 两个 d/2 半步的平均速度（stop-grad，I8），使 4 步 Euler 少步生成可用。
- **reward/cont**（可选）：two-hot symexp NLL / 伯努利 NLL，对齐 context[t]（已见 o≤t, a≤t）
  与转移奖励 reward[t]；仅在线数据可用（离线 VPT 无奖励）。

训练域入口（超参与评估口径见各文件 docstring）：

- **离线** `train/minecraft/train_dreamer4`：VPT/BASALT 真数据（`VPTStreamDataset`，64px，
  22 维连续动作直接进 AdaLN 调制——不要求 one-hot）。评估 = holdout clip 上
  `psnr_gen`（4 步流生成下一帧）对照 `psnr_recon`（重建上限）与 `psnr_persist`
  （复读上一帧基线）：gen 必须逼近/超过 persist 才说明动力学在利用动作。
- **在线** `train/craftground/train_dreamer4`：CraftGround(Minecraft 1.21) 随机探索采集
  交互流，边采集边训练（含 reward/cont 头）；`--init` 从离线 checkpoint 热启动
  （动作接口 22 维连续→27 维 one-hot 不同，action_proj/reward/cont 重新学）。
  评估在 **held-out 环境**（最后一个 env 的数据不进训练）。

想象 actor-critic / 策略蒸馏阶段仍待补（`tests/integration/test_dreamer_build.py`
覆盖连续/VQ 两种 tokenizer 的构建前向）。

### 2.6 混合精度与降精度训练（Minecraft Dreamer4，L4 实测 2026-07-02）

两个 Dreamer4 训练域均有 `--amp {off,bf16,fp16}`（默认 bf16）。同一 62M 配置
（token_dim 384 / dyn_layers 8 / enc_base 48 / batch 24）三精度对照，60 步内
损失轨迹逐位一致（step40 total 均 1.0662）、holdout PSNR 一致：

| 精度 | step40 累计吞吐 | 备注 |
|---|---|---|
| fp32(+TF32) | 667 帧/s | TF32 已默认开启（§2.5），matmul 走 tensor core |
| **bf16 autocast** | **839 帧/s（+26%）** | **默认**。指数位同 fp32 ⇒ 无上下溢、无需 GradScaler |
| fp16 autocast | 843 帧/s | 与 bf16 同速（Ada 上无优势），额外要 GradScaler ⇒ 不选 |

要点与边界（与 I4 的关系）：

- **autocast 是"混合"而非"全低精度"**：权重主本与优化器状态仍 fp32；norm/softmax/
  log/exp/损失 reduction 由 autocast 自动保持 fp32——I4（危险算子 fp32）无需手工处理。
  评估路径不进 autocast（指标口径恒 fp32）。
- **流匹配对 bf16 不敏感的原因**：速度目标 z₁-ε 与 MSE 都是 O(1) 量级、无长链
  数值积累；shortcut 自一致目标本身带 stop-grad（I8），半步误差不回传。
- **进一步降精度的选项（未做，按性价比排序）**：
  1. `torch.compile` 编/解码器：核融合，预计 1.1-1.3×;代价是编译时间与形状变化的
     graph break，形状固定后可开。
  2. **8-bit 优化器**（torchao/bitsandbytes）：优化器状态从 fp32→int8，省约 6 字节/参数
     显存,用于换更大 batch;精度损失有公开基准背书,比"纯 bf16 权重主本"稳妥
     （后者小更新量会被 7 位尾数吞掉,不推荐）。
  3. **fp8（E4M3/E5M2,Ada 原生支持）**：需 Transformer Engine 与逐张量缩放,
     matmul 密集的 dynamics 或再得 1.3-1.6×;但 62M 规模部分是 overhead-bound,
     收益不确定,等模型上到数亿参数再评估。
- **更先进 GPU 上的 FP8/NVFP4**（当前 L4 用不上,记录以备换卡）:
  - **FP8（E4M3/E5M2）**:Hopper(H100)/Ada(sm 8.9,含 L4)张量核原生支持,经
    NVIDIA Transformer Engine 的 `fp8_autocast` + 逐张量/逐块缩放使用。只加速
    matmul 密集层(te.Linear/attention/MLP),卷积 tokenizer 收益小;本仓 62M 规模
    overhead-bound,预期收益 <1.3×,数亿参数后再评估。
  - **NVFP4（Blackwell,B200/GB200/RTX50)**:4 位浮点(E2M1)+ 16 元素微块 FP8(E4M3)
    缩放 + 逐张量 fp32 缩放(区别于 OCP MXFP4 的 32 元素 2 的幂缩放)。GEMM 吞吐约为
    FP8 的 2×。**训练**可行但需完整配方(NVIDIA 2025 预训练报告):随机 Hadamard 变换
    摊平离群值、梯度随机舍入、首末层保高精度、bf16 主权重——不是一个开关,是一套
    数值工程;TE 提供实验性支持。**推理**侧 NVFP4 已较成熟(TensorRT-LLM)。
  - 结论:L4 上 bf16 是甜点;换 H100 → 上 TE FP8;换 Blackwell 且模型到数亿参数
    → 评估 NVFP4 训练配方。任何降精度都保持评估路径 fp32(口径不变)。
- 与**推理量化**（int8/int4 PTQ/QAT,部署用）是两条线,勿混:本节只谈训练精度。
- 提升 GPU 利用率的第一杠杆仍是 **batch**（`scripts/sys_monitor.py` 低于 30% 滑窗
  均值会告警）:bf16 激活减半,同显存可再放大 batch 约 2×。

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

## 4. Crafter 训练手册（L4 实测）

权威来源是代码（`train/crafter/`、`net/dreamerv3/`）；本节补充可复用的运行方法与一轮观测结果。

### 4.1 从零开始训练（任意 Linux + CUDA 平台）

```bash
# 0) 进入项目根目录，后续命令均从根目录执行
cd <repo>

# 1) 安装依赖。requirements.txt 不含 crafter，需单独安装（会附带 opensimplex / ruamel.yaml）
pip install -r requirements.txt
pip install crafter

# 2) 检查 GPU
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 3) 冒烟测试（tiny，约 4k 步，几分钟，验证整条链路无 NaN）
python -m train.crafter.train_dreamerv3 --size tiny --total-steps 4000 \
    --prefill 500 --run-dir runs/crafter_smoke
# 看到 wm/img 损失下降、有 upd= 日志、最后打印"训练完成"即通过

# 4) 正式训练（small，后台运行 + 行缓冲日志）
nohup python -m train.crafter.train_dreamerv3 --size small --total-steps 200000 \
    --run-dir runs/crafter_dreamerv3 > runs/crafter_dreamerv3/train.log 2>&1 &

# 5) 观测
tail -f runs/crafter_dreamerv3/train.log              # 指标
watch -n1 nvidia-smi                                  # GPU（利用率有突发，需多次采样看均值）
```

效率开关（`net/` 不变，仅在训练入口 `_enable_fast_math()` 设置）已默认开启：TF32 + `cudnn.benchmark` + 关闭 `torch.distributions` 参数校验。默认大 batch 短序列：`batch=48 seq=32 updates_per=2 train_every=1`（train ratio 384）。优化的原理见 §2.5。

### 4.2 一轮训练结论（small，跑到 75.8k / 200k 步后手动停止）

- **世界模型：正常收敛**。`wm_total 240 → ~35`、`image 237 → ~26`，单调下降后趋于平台，无 NaN。说明训练管线运行正常、世界模型在学习。
- **策略：40k 步到峰后回落**。`ep_rew` 从约 1.0 升到峰值约 2.4（@40k），随后回落至约 1.8（@75k）；`ach/ep` 峰值约 3.3（@40k）回落至约 2.7。actor 熵从约 2.3 降至约 1.0，策略逐渐收窄，在有限预算内没有突破更难的成就。
- **最佳检查点：`checkpoints/ckpt_00040000.pt`**（不是最新的）。评测、可视化、热启续训均建议使用此检查点。
- **参照尺度**：Crafter 原生奖励 = 每首次解锁成就 +1，加生命值 ±0.1 塑形。随机基线约 2.1；DreamerV3 在完整 1M 步预算下约 10–11（成就 8–9）。本轮 200k 步是完整预算的 1/5，加之 40k 后策略退化，绝对分偏低属正常。

### 4.3 并行环境数（n_envs）与 GPU / CPU 的关系

`VecCrafterEnv` 在单进程内顺序步进 n_envs 个 Crafter 实例。Crafter 世界生成是纯 Python，受 GIL 约束，因此串行 env 步进的墙钟时间与 n_envs 成正比，增加 CPU 核数不会自动加速。`env.py` 另提供 `SubprocVecCrafterEnv`（子进程并行，`train_ppo_ad` 默认启用）；`train_dreamerv3` 当前仍用串行 `VecCrafterEnv`。

| 资源 | 负责 | 受什么主导 |
|---|---|---|
| **CPU** | 环境步进（串行）+ RSSM `observe` 沿 T 的逐步调度 | n_envs、seq_len；是实际吞吐上限（overhead-bound） |
| **GPU** | 世界模型前向/反向 + 想象 actor-critic | batch × seq × updates_per；与 n_envs 无关 |
| 关系纽带 | `train ratio = updates_per × batch × seq /(train_every × n_envs)` | 调此值平衡数据供给与墙钟时间 |

**n_envs 建议**：
- **当前串行 env（train_dreamerv3）**：`n_envs=8` 较优（配合 train ratio ≈ 384/512）。超过 16 基本只会更慢，不建议。
- **更强 GPU（A100/H100）**：优先增大 batch（48→64–96，RSSM overhead 与 batch 几乎无关），或把 `--size` 从 `small` 改为 `default`；n_envs 仍保持 8–16。显存参考：B=64、T=48 峰值约 20GB（24GB 卡安全上限），B≤64、T≤32 < 14GB。
- **利用多核 CPU**：改用子进程并行 env（`SubprocVecCrafterEnv`），n_envs ≈ 物理核数，env 吞吐可接近线性扩展。

### 4.4 评测 / 续训

- **评测**：`ckpt_00040000.pt` 含 `{"total_steps", "model_state", "ep_rewards"}`；用 `net.dreamerv3.build_dreamerv3(..., **SIZE_PRESETS["small"])` 重建模型后 `load_state_dict(ckpt["model_state"])`，再用 `agent.policy()` 在 `VecCrafterEnv` 上跑 rollout 渲染帧或统计成就。
- **续训**：训练脚本目前**没有 `--resume` 入口**（只存检查点，不读取）。两条路：① `--total-steps 1000000` 从头重训（L4 约 1 天，建议更强的卡 + 更大 batch）；② 给 `train_dreamerv3.py` 增加加载检查点的 `--resume`，从 `ckpt_00040000.pt` 热启（需改代码）。针对策略退化，可提高 actor 熵正则系数或调 `--ac-lr` 缓解熵过早塌缩。
