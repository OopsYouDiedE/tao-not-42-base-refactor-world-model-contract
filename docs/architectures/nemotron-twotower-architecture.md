---
name: nemotron-twotower-architecture
description: Nemotron-Labs-TwoTower 数学分析:冻结 AR 塔 + 可训块扩散塔;含移植到 dreamer4 做"换游戏快速适应"的对应关系
metadata:
  type: reference
---

# Nemotron-Labs-TwoTower 架构分析(学术研究对象)

**论文**: arXiv:2606.26493《Nemotron-TwoTower: Diffusion Language Modeling with Pretrained Autoregressive Context》(2026-06/07)
**权重**: HF `nvidia/Nemotron-Labs-TwoTower-30B-A3B-Base-BF16`(仅 Base,无对齐版)
**定位(本项目)**: "冻结通用塔 + 可训读者塔"配方的规模化实证——移植目标是 dreamer4 的换游戏快速适应

## 1. 结构:同源双塔

两塔均为 Nemotron-3-Nano 52 层混合骨干(23 Mamba-2 + 6 GQA + 23 MoE)的完整副本,
**同一预训练 checkpoint 初始化**,合计 ~60B 总参 / 每塔激活 ~3B:

- **Context 塔(冻结)**:因果 AR,读 prompt + 已提交 token,产出逐层 KV + Mamba 终态;
- **Denoiser 塔(可训)**:块内掩码扩散,一次去噪一整块(采样默认 S=16)。

## 2. 数学

### 2.1 训练目标(掩码块扩散)

```
L_MD = E_{t, z_t} [ 1/|M_t| · Σ_{(b,ℓ)∈M_t} −log p_θ(x_b^ℓ | z_t^b, t, c_{<b}) ]
```

线性噪声表 α_t = 1−t:每 token 以概率 1−α_t 独立替换为 [MASK];对被掩位置取平均 NLL
(**弃掉理论上的 1/t 重要性权重以稳训**)。时间条件用 adaLN-single:全局 MLP 出共享
scale/shift/gate + 逐层可学嵌入,只增 1.5M 参数——与本仓 dreamer4 动作 AdaLN 同习语。

### 2.2 跨塔交互(嵌合的精确形式)

**层对齐 cross-attention**(第 i 层查第 i 层):

```
Attn(Q_b^(i), [K_{<b}^{ctx,(i)} ; K_b^{den,(i)}], [V_{<b}^{ctx,(i)} ; V_b^{den,(i)}])
```

块内噪声 token 双向互看,对已提交前缀因果地查**冻结塔**的 KV。
**Mamba 状态播种**:Denoiser 各 Mamba-2 层的初始状态 = 冻结塔在块 b−1 结束时的对应层状态。
即"过去"以双通道递交:KV(精确、随长度增长)+ Mamba 终态(压缩、常数大小)。
同源初始化保证第 i 层天然读得懂第 i 层——跨塔无需翻译层。

### 2.3 采样(置信度逐步揭示)

每步并行预测全部掩码位,置信度 > γ=0.8 的提交、其余留待下轮;首步提交最多,
且呈"左上三角"模式(继承自 AR 骨干的从左到右偏置)。块间仍严格自回归。

## 3. 关键实验事实

- **质量/速度**:综合基准保持 AR 基线 98.7%,墙钟吞吐 2.42×(2×H100 BF16,端到端计时);
- **训练成本**:仅 1.4T token(骨干预训练 25T 的 ~6%)——"改装"比重训便宜一个量级;
- **块大小**:训练 S=32→16 分阶段;**采样超训练块长即崩**(HumanEval S=16:76.4 → S=64:19.85);
- **双塔解耦消融(Table 2,对本项目最重要)**:
  - 冻结 Context + 独立 Denoiser:基线(最优配置);
  - 联合微调/绑权重共损失:精度 −26~27%——**冻结不是妥协,是赢家配置**;
  - 双向化 Mamba:无收益(72.94→72.96%),保持因果即可。

## 4. 对本项目的接口含义:移植到 dreamer4 做换游戏快速适应

配方:**冻结 gaming500 预训练的通用动力学塔 + 每游戏一个可训增量塔**。对应关系与需改处:

| TwoTower | dreamer4 | 需改? |
|---|---|---|
| 离散 token + [MASK] 吸收态腐蚀 | 连续 DINO token + 高斯插值 x_τ=(1−τ)ε+τz₁ | 不改——流匹配保留 |
| 掩码位 NLL(分类) | 速度场 MSE(ShortcutHead 已有) | 不改 |
| 置信度逐步揭示 | shortcut 少步 Euler 积分 | 不改(连续域无"提交"概念,少步采样即其对应物) |
| 块 = 16 文本 token | 块 = 一帧 121 token(本就整帧并行去噪) | 不改——dreamer4 天然是"单塔 TwoTower" |
| 层对齐 cross-attn 查冻结塔 KV | **需新增**:增量塔各层对冻结塔同层 token 网格 [B,T,S,D] 的 cross-attention(blocks.PreLNAttn 支持 cross) | 主要工程点 |
| Mamba 状态播种 | dreamer4 时间混合是因果注意力,无递归状态——"播种"退化为"共享冻结塔时间层 KV"(即上一行的 cross-attn 本身) | 合并处理 |
| adaLN-single 时间条件 | 动作 AdaLN(零初始)已有,τ/d 条件在 ShortcutHead 已有 | 不改 |

可直接继承的三条实验教训:
1. **冻结通用塔 + 独立增量塔 > 联合微调**(Table 2 差 26%)——为"换游戏只训读者塔"
   的快速适应论提供了 30B 级证据;
2. 增量塔可从冻结塔**同权重初始化**(同源方言),小数据新游戏上收敛起点即通用物理;
   参数预算紧可退化为 LoRA 化增量塔;
3. 推理期双塔 ⇒ 参数 ×2:对 60B LLM 是代价,对我们 ~10⁸ 级动力学塔可忽略。

风险标注:TwoTower 的冻结塔与增量塔**吃同一分布**(文本);我们的冻结塔是旧游戏分布、
增量塔吃新游戏——分布位移下"层对齐可读"是否保持,是移植实验的第一个待证闸门
(可用 Gate 1 的 IG 口径:增量塔相对"从零小塔"的信息增益)。
