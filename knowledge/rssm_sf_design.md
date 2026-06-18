# RSSM + 后继特征(Successor Features)世界模型设计

> 本文档解释**为什么这样改**与**数学上为什么成立**(SSOT:微观实现以 `net/rssm.py`、
> `train/minecraft/train_rssm.py` 代码为准,本文不复述实现细节)。

## 1. 动机:三个正交退化 → 三个结构件

旧的"契约预测器"(`net/world_model.py` 的 anchor + 有界增量 + 一堆一次性损失)在 holdout 上
只能勉强追平 copy-last,且"后果感知"不可学。根因是三个**互相独立**的退化,各缺一个结构件:

| 退化 | 现象 | 缺的结构件 |
|---|---|---|
| **D1 坍缩** | z_inv 朝 N(0,1) 掉信息,`w_CoV→0` | 预测性先验 + free-bits KL(保信息) |
| **D2 不可达** | `anchor=最近帧`+有界增量,多步未来不在可达集 | 递归状态 h_t(开环可滚) |
| **D3 无信用算子** | 全是一次性损失,长程后果无通道 | TD/Bellman 递归(后继特征 ψ) |

三件互不冲突,都是"把表征钉向要紧信息"的 grounding 力,可叠加。

## 2. 架构总览

帧级 RSSM(DreamerV2/V3 谱系),decoder-free,用**冻结骨干嵌入**作 grounding 目标:

```
img_t ──冻结DINO+池化──> e_t (固定嵌入, 不参与梯度, 不会坍缩)
                          │
状态 s_t=(h_t, z_t):
  确定递归  h_t = GRU([z_{t-1}, a_{t-1}], h_{t-1})       # 把全历史积分进来
  先验      p(z_t | h_t)        (不看观测, 可开环想象)    # D2: 多步可达
  后验      q(z_t | h_t, e_t)   (看观测嵌入)
  随机态    z_t = (z_rev 高斯连续, z_inv 离散组)          # 保留 rev/inv 因子化
feat_t = [h_t, z_t]
  grounding 头   ê_t = g(feat_t) ≈ e_t   (MSE, decoder-free 重建)  # D1
  后继特征头     ψ_t = ψ(feat_t)  由 TD(λ) 拟合                    # D3
```

旧的 `latent_align / agree / guide / null / event_ce` 整体退役——它们想做的事被
`KL(q‖p)`(预测性先验)与 `ψ`(长程后果)在原理上替代。保留 SIGReg 思想由 free-bits KL +
categorical unimix 接管防坍缩。

## 3. 数学:为什么每件成立

### 3.1 预测性先验 + free-bits KL(治 D1)
变分下界 `log p(e) ≥ E_q[log p(e|z)] − KL(q(z|h,e) ‖ p(z|h))`。decoder-free 下
"`log p(e|z)`" 即 grounding 头对**固定**嵌入 e 的预测(目标冻结 ⇒ 重建项无坍缩平凡解)。
最小化 KL 即逼先验 `p(z|h)` 学会"不看图也能预测下一潜分布"——这正是旧模型缺的开环预测力。

- **free bits**:`max(KL − τ, 0)`。`KL<τ` 处梯度恒 0 ⇒ τ nats 内携带信息免费,
  **可证移除 posterior collapse 吸引子**(Kingma 2016;DreamerV2/V3 标配)。
- **KL balancing**:`dyn=KL(sg(q)‖p)` 训先验、`rep=KL(q‖sg(p))` 训后验,
  `dyn_scale>rep_scale` ⇒ 先验向后验靠得更快,阻止先验靠抬自身熵迁就懒后验。

### 3.2 后继特征 ψ 是 γ-压缩不动点,TD 可学(治 D3,本设计的核心)
对每个自我结局特征 `φ`(本切片 φ=has_item),定义后继特征
```
ψ^π(s) = E_π[ Σ_{k≥0} γ^k φ(s_{t+k}) ]
```
它满足 Bellman 方程 `ψ^π(s)=φ(s)+γ·E[ψ^π(s')]`,**每一维一个 γ-压缩算子**
(sup-norm,Banach ⇒ 唯一不动点、TD 收敛)。ψ 对第 k 步结局的权重是 `γ^k`,
horizon ≈ `1/(1−γ)`:取 γ≈0.997 即按定义编码约 300 帧后的自我结局。

为什么用 ψ 而非单标量 value:ψ **reward-agnostic、稠密、向量值**——它直接是
"在此状态下,每种结局未来折扣发生量"的指纹,正是"感知几百帧后对自己的影响"的数学形式;
任意奖励 `r=w·φ` 立得 `V=w·ψ`,但不必先选奖励。稠密性根治稀疏奖励(每帧每维都有 TD 目标)。

### 3.3 递归态 ⇒ 多步可达(治 D2)
`h_{t+k}` 由先验开环滚动得到,可达集无"锚球"封顶 ⇒ 远期未来可表达。
旧模型 `anchor⊕B(r)` 的有界球随 horizon 必被真实未来流形甩开;递归滚动消除此上界。
误差累积(`~L^k`)由"按多步先验 rollout 训练 / latent overshooting"压制,
由验收线 1(难 horizon)和 rollout 漂移监控。

## 4. 离线数据的根本天花板(必须标清)

离线 VPT 日志没有干预,只能拟合**行为策略 μ 的** `ψ^μ`——
「按人类这么玩,未来结局期望」= **相关性后果**,不是「*我* do 这个动作给*自己*带来什么」的
**因果**自我影响。要因果:接环境交互闭环,或在 ignorability/overlap 强假设下做反事实辨识。
`ψ^μ` 已远超旧模型(第一次真有长程后果通道),但它是终点前一站。与
[claims_and_scope.md](claims_and_scope.md) §10「主动交互轴未测」一致。

TD 在此是 **on-policy 评估 μ**(沿 μ 自己的数据评估 μ),落在 deadly triad 的良性角
(不做 off-policy max),配 slow-target 更稳。

## 5. 本切片范围 vs 留待

**本切片已含**:帧级 RSSM(GRU + 先验/后验 + free-bits balanced KL)、rev/inv 因子化随机态、
decoder-free grounding 头、单维后继特征 ψ(φ=has_item)+ TD(λ)、两条可证伪验收线。

**留待(本切片不做)**:γ→0.997 + 更长序列上的几百帧 horizon;多维 ψ(全 φ);
symlog/two-hot 头与 RewardEMA 归一化;actor(离线不可信,需环境);骨干部分解冻(探针失败再开)。
切片目标是**证明机制成立**(难 horizon 能赢 copy-last + ψ 有 dose-response),非生产规模。

## 6. 两条可证伪验收线

1. **难 horizon align_ratio < 1**:后验观测到 k=T//2,用动作把先验开环滚到 T−1,
   grounding 头解码 ê_{T−1};`ratio = ‖ê_{T−1}−e_{T−1}‖² / ‖e_k−e_{T−1}‖²`。
   `<1` = 在旧模型必败的难 horizon 上真赢 copy-last。
2. **ψ dose-response**:holdout 上 `corr(ψ_t, D_t)`,其中
   `D_t = Σ_k γ^k φ_{t+k}`(经验折扣未来发生量)。`>0` 且 train≈holdout(不过拟合)、
   分位分箱单调递增 = ψ 真感知到了未来结局,而非记忆。

## 7. 文件清单

- `net/config.py::RSSMConfig` — 类型化结构 schema。
- `net/rssm.py` — RSSM 核 + grounding 头 + 后继特征头 + free-bits balanced KL(纯 net/,无 IO/domain/mock)。
- `train/minecraft/train_rssm.py` — 冻结骨干感知包装 + 训练步 + 两条验收指标 + CLI。
- `tests/integration/test_rssm.py` — CPU mock 嵌入冒烟(observe/imagine/KL/SF 前向反向有限,两验收线可算)。
