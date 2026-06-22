# Dreamer 系世界模型 — 设计与放置说明

> 本文档管 **Dreamer 系**(DreamerV3 + Dreamer4)的宏观设计与目录放置。SSOT:数值逻辑以
> `net/dreamerv3/`、`net/dreamer4/`、`blocks/` 代码为准,本文只解释架构意图与"为什么这么放"。

## 0. 现状(2026-06,清白重建)

仓库早先 vendored 过一份完整的 DreamerV3(NM512/dreamerv3-torch,MIT),后整目录退役删除,
**仅保留被 `blocks/` 复用的独立算子**:`GRUCell`(blocks/dynamics.py)、`Conv2dSamePad`/
`ImgChLayerNorm`(blocks/conv.py)、`ConvEncoder`/`ConvDecoder`(blocks/encoder.py、decoder.py)、
symlog·two-hot·OneHot 等分布(blocks/distributions.py)、`static_scan`·`lambda_return`
(blocks/sequence.py)。MIT 署名/许可证见 `blocks/NOTICE.dreamerv3`、`blocks/LICENSE.dreamerv3`。

当前的 `net/dreamerv3/` 与 `net/dreamer4/` 是**从 `blocks/` 算子库清白重建**的实现
(**非 vendored、非逐字照抄**):L2 网络层把那些与任务无关的算子组装成两个完整的世界模型。
故二者受本仓代码规范(I1–I8、写作纪律、依赖方向 blocks←net←train)约束,与旧 vendored 目录不同。

## 1. DreamerV3(`net/dreamerv3/`)— 可训练基线

Hafner 等《Mastering Diverse Domains through World Models》(arXiv:2301.04104)的结构。
训练任务:Crafter(64×64 RGB,Discrete(17))。

| 文件 | 内容 |
|---|---|
| `config.py` | `DreamerV3Config` 纯 dataclass(RSSM/编解码/头/想象的结构超参,默认对齐上游 defaults) |
| `rssm.py` | `RSSM`:`GRUCell` 确定性 deter + 离散 32×32 随机隐变量(`OneHotDist` unimix);`observe`/`imagine_with_action`(`static_scan` 展开)/`kl_loss`(动力学/表征 KL + free-bits) |
| `world_model.py` | `WorldModel`:`ConvEncoder` + `RSSM` + `ConvDecoder`(`MSEDist` 重建,图像在 [-0.5,0.5])+ reward(`DiscDist` two-hot symexp)+ cont(`Bernoulli`)头;`loss()` 给重建+奖励+终止+KL |
| `behavior.py` | `ImagBehavior`:想象 rollout 上的离散策略(`OneHotDist`)+ two-hot 价值(`DiscDist`)+ 慢靶 critic;`lambda_return` 算价值目标,`RewardEMA`(5/95 分位)归一优势,reinforce 策略梯度。rollout 与回报目标全程 `no_grad`(reinforce 不需路径梯度 ⇒ 省显存且不改学习信号) |
| `agent.py` | `DreamerV3`(持有 wm+behavior,`policy()` 递归单步交互)+ `build_dreamerv3(**overrides)` 工厂 |

依赖方向遵约:`net/dreamerv3/` 只 import `blocks` 与本包,**不含训练循环/优化器/数据加载**。
训练循环在 `train/crafter/`:`dreamer_buffer.py`(`SequenceReplay` 定长序列回放,CPU/uint8 存储)+
`train_dreamerv3.py`(采集 ↔ 世界模型更新 + 想象 actor-critic 更新;wm 与 actor/critic 各自优化器,
想象损失不回传梯度到世界模型)。`--size {tiny,small,default}` 选结构规模。

**验证(L4 GPU,tiny,6k 步冒烟)**:世界模型损失单调下降(image 429→179、reward 4.1→0.5、
wm_total 434→181),KL 稳定 ~1.6,无 NaN;采集→wm 训练→想象→actor-critic→递归 policy 全链路跑通。
该规模/步数下策略尚未充分提升(actor lr 3e-5、仅数百次更新),世界模型学习信号是当前的明确正向证据。
CPU 小尺寸前向+反向冒烟见 `tests/integration/test_dreamer_build.py`。

## 2. Dreamer4(`net/dreamer4/`)— 仅构建,暂不训练

Hafner 等《Dreamer 4: Training Agents Inside of Scalable World Models》(2025)的结构骨架,
从 `blocks/` 组装。**本仓只构建、不提供训练循环**(流匹配世界模型训练 + 想象 actor-critic 待补)。

| 文件 | 内容 |
|---|---|
| `config.py` | `Dreamer4Config`(tokenizer/动力学/shortcut 头/各头的结构超参;为本机可构造做了缩放,非论文原尺度) |
| `tokenizer.py` | `Tokenizer`:`ConvEncoder`(flatten=False 取空间特征图)→ 每空间位置一个连续潜 token;可选 `VectorQuantizer` 离散码本瓶颈;`ConvDecoder` 还原图像 |
| `dynamics.py` | `SpaceTimeTransformer`:每块 = 帧内空间自注意(`MHABlock` 非因果)+ 跨帧因果时间自注意(`MHABlock` causal)+ 动作 AdaLN 调制(零初始恒等);`ShortcutHead`:shortcut-forcing 流匹配速度头(给定上下文+噪声 token+流时间 τ+步长 d → 速度 v,支持少步生成) |
| `world_model.py` | `WorldModel`:tokenizer + 动力学 + shortcut 头 + reward/cont 头;`forward()` 跑一次形状自洽前向(编码→上下文→少步 Euler 流生成→解码) |
| `agent.py` | `Dreamer4`(wm + actor/critic 头)+ `build_dreamer4(**overrides)` 工厂 |

为何只构建:Dreamer4 的核心训练目标(shortcut forcing 的流匹配 + 一致性、想象 actor-critic)
与 DreamerV3 差异大,需要单独的训练域实现;当前阶段先把结构按 blocks 分层落地、跑通 shape 契约
(`tests/integration/test_dreamer_build.py` 覆盖连续/VQ 两种 tokenizer 的前向),训练循环后续按需补。

## 2.5 训练实践笔记(Crafter + L4)

落盘的操作性经验(原理性、可复用,非活动流水账):

- **吞吐受环境步进主导,GPU 呈突发利用**:Crafter 在 CPU 上顺序步进 n_envs 个实例(纯 Python
  世界生成,`VecCrafterEnv` 为 Colab 兼容不用多进程),env 步进是 wall-clock 瓶颈;单次世界模型
  前向+反向(small=6.7M 参数、batch 16×64)很快,故 GPU 利用率是**突发**的——
  单次 `nvidia-smi` 快照常落在两次突发之间的 env 步进间隙,读到偏低甚至 0,需多次采样看均值
  (实测 small 配置约 20–80% 突发、显存 ~5.8/23 GB)。
- **抬高 GPU 利用率 / 样本效率的旋钮是 train ratio**:即 `updates_per`(每次环境迭代的梯度步数),
  其次是 batch/seq/模型规模。本仓 train ratio = `updates_per × batch × seq /(train_every × n_envs)`。
  DreamerV3 原版经典 train ratio ≈ 512(回放帧 / 环境步);`updates_per=4, train_every=1, batch=16,
  seq=64, n_envs=8` ⇒ 512,正好对齐。更大的 train ratio 提样本效率与 GPU 占用,但增 wall-clock。
- **回放放 CPU、采样转 GPU**:`SequenceReplay` 以 uint8 在 CPU 存 obs(节省显存),采样的小批量
  才转设备;容量 non-wrapping,按 `total_steps/n_envs` 预留,用尽即停止写入(避免环形窗口跨写指针)。
- **想象 rollout 全程 `no_grad`**:reinforce 策略梯度不需沿 rollout 的路径梯度(actor 仅由显式
  logπ·advantage 接收梯度,critic 在 detach 特征上回归),切断 rollout 图省显存且不改学习信号。
- **RSSM 因果对齐**:训练 `observe` 必须喂右移一位的 `prev_action`(进入 obs[t] 的动作 = action[t-1]),
  与想象 `img_step(state, departing_action)` 的因果方向一致;否则训练用"离开当前帧的动作"预测当前状态、
  与想象不一致,会污染想象 rollout 动力学、损害策略学习(见 world_model.py 注释与 git fix)。
- **日志缓冲**:`nohup … > log` 重定向时 Python stdout 默认全缓冲,进度日志长时间不落盘会让进程
  看似卡死;`train_dreamerv3` 已在 `main()` 开头设 stdout 行缓冲(等价 `python -u`)。

## 3. 升级/借鉴关系

- DreamerV3 的随机隐变量(离散 32×32)是本仓统一世界基座里 ξ 思想的来源(见 mental_world)。
- 两个模型共享 `blocks/` 的同一批算子(GRU/卷积/分布/序列/注意力/MLP),互为对照:
  DreamerV3 用 GRU 递归 + 离散 RSSM,Dreamer4 用时空 Transformer + 连续 token + 流匹配生成。
