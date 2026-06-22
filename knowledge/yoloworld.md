# YOLO-World-Dreamer 设计(net/yoloworld)

> 在 DreamerV3 世界模型基座上,把 **YOLO26 端到端候选 + YOLOv10 双头分配 + YOLOE 文本点乘**
> 三个思想嫁接成一个**目标条件**(goal-conditioned)规划器。本文只记宏观架构与算法意图;
> 形状/超参的唯一事实来源是 `net/yoloworld/config.py` 与各模块 docstring(SSOT)。

## 1. 一句话

世界模型照常用真实序列(长 16/32)自监督训练;在其隐状态上挂**两条行为线**——
一个便宜的 **256 候选动作序列小头**(采集动作时只跑它,端到端、无搜索)与一个昂贵的
**rollout 老师头**(把候选喂进世界模型想象、算回报、给小头当监督)。任务用语言描述并编码,
小头的每条候选嵌入与任务编码**点乘**直接选出最优序列(YOLOE)。

## 2. 记号

| 符号 | 含义 | 形状 |
|---|---|---|
| `s=(h,z)` | RSSM 隐状态(deter+stoch) | `h:D`, `z:S×C` |
| `φ=[flat(z),h]` | 世界状态特征 | `d_φ=S·C+D` |
| `a` | 离散动作 | one-hot `A=17` |
| `g=enc(ℓ)` | 任务语言冻结句向量(MiniLM) | `d_g=384`(单位球) |
| `E` | 22 句成就描述的冻结嵌入矩阵 | `U×d_g`,`U=22` |
| `H` | 规划/想象步长 | 8–16 |
| `K` | 候选动作序列数 | 256 |
| `M` | rollout 老师实际滚动的候选数(top-M) | 32 |

## 3. 目标函数(语言点乘导出的目标条件价值)

每 episode 采样目标成就 `u`,语言 `ℓ(u)→g`。世界模型加一个**成就预测头** `ψ(s)∈[0,1]^U`
(标签 = 该步累计已解锁成就的 multi-hot)。任务对成就的权重与"完成度"势函数:

```
w(g) = softmax(E·g / τ) ∈ Δ^{U-1}
ρ^g(s) = w(g)·ψ(s) ∈ [0,1]          # YOLOE 在「状态层」的点乘
```

**势函数塑形奖励**(Ng et al.,策略不变、稠密,可在想象内自算):

```
r^g_t = γ·ρ^g(s_t) − ρ^g(s_{t-1})
```

目标:`J(π;g) = E[ Σ_t γ^{t-1} r^g_t ]`。

> 设计决策(已确认):用势函数塑形而非原始稀疏成就奖励;rollout 老师只滚 top-M 候选而非全 256。

## 4. 256 候选 = 对不可解 argmax 的稀疏摊销近似(YOLO)

序列动作价值(**rollout 头**的估计量):

```
Q^g(s, a_{1:H}) = E[ Σ_{τ=1}^H γ^{τ-1} r^g_τ + γ^H V^g(s_H) ]
```

`argmax_{a_{1:H}} Q^g` 在 `A^H=17^16` 上不可解。小头一次前向用 `K=256` 条可学候选摊销它:

```
{(π^k, p^k, e^k)}_{k=1..K} = Head_ξ(φ(s), g)
```

`π^k` 第 k 条计划的每步动作分布,`p^k` 标量 logit,`e^k∈S^{d_g-1}` 计划嵌入。诱导稀疏混合分布:

```
α^k = softmax(p^k + β·(e^k·g))_k          # YOLOE 点乘决定混合权重 = 选择信号
π_ξ(a_{1:H}|s,g) = Σ_k α^k Π_τ π^k_τ(a_τ)
```

## 5. 两条线的损失(从单一目标导出双头一致性)

**老师(rollout 头,one-to-many,重算)**:对 top-M 候选用 WM 先验滚 H 步得 `R^k=Q^g(s,a^k)`,
软信念 `t^k = softmax(R^k/η)_k`。

| 损失 | 公式 | 作用 |
|---|---|---|
| `L_cls`(YOLOv10 一致性) | `KL(t ‖ α)` | 小头混合权重逼近老师排序;η→0 时退化为 one-to-one(top-1 = rollout argmax) |
| `L_plan`(REINFORCE+群体基线) | `−Σ_k Σ_τ (R^k−R̄) log π^k_τ(a^k_τ) − λ_H·H[π]` | 256 候选互为蒙特卡洛基线,方差自削减 |
| `L_align`(YOLOE 对齐) | `Σ_k (1 − e^k·sg[ê^k_roll])` | 小头无 rollout 即可预测计划达成的成就嵌入 → 对目标开放词表 |
| `L_div`(slot 互斥) | `‖Ḡ − I‖²_offdiag`,`Ḡ=ⵧ̄·ⵧ̄ᵀ`,`ⵧ̄`=batch 平均 slot 嵌入 | 反候选坍缩:K 个 slot 在 d_g 空间近正交 → **不同 slot 不同动作语义** |
| `L_load`(使用均衡) | `Σ_k ᾱ_k log ᾱ_k`(batch 平均选择负熵) | 反坍缩:均衡 slot 使用,逼不同状态选不同 slot → slot 专精 |
| `L_critic` | two-hot symexp 回归想象 λ-return + 慢靶 | 提供 `Q` 的 bootstrap `V^g(s_H)` |
| `L_WM`+`L_ach` | DreamerV3 ELBO + 成就头 BCE | 世界模型线 |

老师计划嵌入:`ê^k_roll = normalize(Σ_τ γ^{τ-1} Eᵀ ψ(s^k_τ))`。

**总损失**(行为线对 WM 特征 stop-grad):

```
L = [L_WM + L_ach]            ← 世界模型线
  + [L_cls + L_plan + L_align + L_critic]   ← 行为双头线
```

## 6. 结构与尺寸

**小头(DETR 式 256 query + 共享解码器)**:query 表 `[K,d_q]`;任务投影 `g→g̃∈R^{d_g'}`;
上下文 `c=MLP([φ,g̃])`;每 query `x_k=[Q_k,c]` → 共享 MLP → 计划 logits `[H,A]`、`p^k`、`e^k`(L2 归一)。
参数 ~10^5,算力可忽略。

**rollout 老师**:几乎无新参数(复用 RSSM + ψ + critic),算力在 `B·M·H` 次 `img_step`。
算力旋钮:按 α 预排序只滚 top-M + 少量 ε 随机候选保覆盖;`n_start` 子采样起点。

**critic** `V^g(φ,g̃)`:two-hot symexp + 慢靶(沿用 dreamerv3 behavior 思路)。

**尺寸预设**(`config.py`):`crafter` = 能学会 Crafter 的 DreamerV3 已验证档(deter=512、stoch 32×32、
units=512、conv (32,64,128,256),~1.7×10⁷ 参数);`small`/`tiny` 供 CPU 冒烟。

## 7. 采集动作(环内无 rollout)

`obs_step` 得 `s_t` → 小头一次前向 → `k̂=argmax_k(p^k+β·e^k·g)` → 执行 `π^{k̂}_1` 首动作 →
下步重规划(receding-horizon)。控制环只有一次小头前向 + 一次 WM `obs_step`,这是 YOLO26 端到端省算点。

## 8. 边界与算力诚实声明

- CPU(Colab)只能跑通正确性 + 小规模出信号;真正训到 Crafter 分数需同份代码挂 GPU
  (`--size crafter --device cuda`,~1M 步)。CPU 上 1M 步是数周量级,是算力问题非代码问题。
- 任务文本编码器冻结(MiniLM):4→22 个成就语义句撑起"任务条件 + 语义空间接口",
  开放指令跟随能力取决于成就句之间是否有真实行为方差。
