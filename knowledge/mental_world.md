# 脑内世界（Mental World）设计

> **文档定位（SSOT）**：本文只讲**宏观架构、算法意图、训练目标的设计动机与物理直觉**。
> 易变的实现细节（精确 Shape/Dtype、层数、超参）归代码本身：
> 活模型实现见 [net/world_model.py](../net/world_model.py)（部件 [slots.py](../net/slots.py) /
> [backbone.py](../net/backbone.py) / [heads.py](../net/heads.py)），积木见 [blocks/primitives.py](../blocks/primitives.py)，
> 数据契约归 [domains/minecraft/vpt_action.py](../domains/minecraft/vpt_action.py)，
> 训练循环与损失见 [train/minecraft/](../train/minecraft/)。
> 本文是设计锚点，记录"为什么这么搭"，不记录历史活动（历史活动归 git log 与项目记忆）。

> ⚠️ **愿景 vs 现状（读法）**：本文同时描述**已落地的活模型**与**未落地的设计愿景**，二者逐节标注。
> - **现状**（§3–§5、§8）= [net/world_model.py](../net/world_model.py) 的 **Minecraft Δz-JEPA 世界模型**，已实现。
> - **方向**（§6）= 当前北极星"看视频掌握玩法"，部分脚手架已就位、核心命题待验证。
> - **愿景**（§7）= 选择性读取（中央凹 / 注视 / 唤醒），**当前无对应代码**（原 `tao_not_42.py` 已删）。
>
> 读"代码现状"以 `net/world_model.py` 为准；读"为什么这么搭、往哪走"以本文为准。
> 读"当前能诚实主张什么、离元愿望多远、值不值得继续"以 [claims_and_scope.md](claims_and_scope.md) 为准(期望-现实校准)。

---

## 0. 一句话立场

把系统从"看视频→反应"改造成**脑内世界**：脑子里维护一组**有限的持久潜向量**，它本身就是一个能自转的生成模型；
眼睛**只在偶尔扫一眼时注入预测误差**去纠正它。框架是 **Transformer**，**弃用 Mamba**。

弃用 Mamba 的理由：当初上 Mamba 的唯一动机是"逐像素 SSM 建立不了运动对应"，其前提是"世界状态 = 像素网格特征图"。
一旦核心状态改为**有限抽象潜向量**（N 很小），该前提消失，Transformer 的全局注意力在小 token 集上更自然，
Mamba 的长序列线性扫描优势也用不上。

---

## 1. 核心表示：有限持久潜向量

- 世界状态 `Z ∈ [B, N, d]`：N 个**跨时间持续存在的潜向量**，不是稠密像素场。以最少数量拟合任务目标。
- **残差/渐进分解（非物体中心 slot）**：slot1 主成分、slot2 补 slot1 漏的、slot3 再校正……每个潜向量装"前面没装下的"。由此带来四条约束：
  - 天然有**幅度谱**（slot1 大 → slotN 小）；slot 间是**顺序正交（Gram-Schmidt 式）**，不是两两对称互斥；
  - **无实体身份可匹配**——Sinkhorn/匈牙利那套 slot 分派**不适用**（那是物体中心 slot 的解药）；
  - 重要性排序由 **nested dropout / Ordered Autoencoder** 思想诱导（随机只留前 k 个 slot 仍要求可重建），不手搓正交损失；
  - ⚠️ **SIGReg 接线戒律**：勿把 `Z` 摊成 `[1, B*N, d]` 做池化 SIGReg——大小尺度混合在 SIGReg 眼里是**非高斯（峰态异常）**，会施压**碾平幅度谱**。须**方向/幅度拆开**：对方向（单位化后）防坍缩，让幅度自由成谱。
- 残余的"多个 slot 冗余绑同一区域"由**结构 + 损失**两层收：结构上 `SlotBinder(compete=True)` 用 `SlotCompetitiveAttn` 沿 **slot 维**做零和竞争（一个 patch 的注意力质量被各槽瓜分）；损失上 `slot_diversity_loss`（`--beta_div`）软惩罚竞争注意力图的成对重叠。约束加在**注意力空间（谁看哪里）**，前向注意力仍是合法分布；**不**在前向里硬正交化（施密特那样硬改聚合权重会打破 `out=w·v` 的加权平均语义、并按 slot 序饿死后排槽）。
- "每次更新完全保持潜向量"= **结构上**潜向量身份/数量不变（持久）+ **算子上**每步只做受限增益的残差改写（非扩张，I5）。
- 范式：**预测编码 / 主动推断**，落成 **predict–correct 两相**（类 Kalman / RSSM）。

---

## 2. 与 LeWM 的区别（本设计的身份证）

本设计**借用** LeWM（github.com/lucas-maes/le-wm）的内核，但在六处刻意偏离。每次"这不就是 LeWM 吗"的疑问，大概率撞上了共享内核——撞对了。

**共享内核（= LeWM，已验证，照搬不创新）**：JEPA 潜空间预测（不解码回像素）+ SIGReg 防坍缩 + Transformer + 全观测提供稠密 target。

**六条真区别（我们的下注）**，并标注当前落地状态（已落地 = 活模型已实现；部分 = 部分维度落地；愿景 = 无对应代码）：

| # | 维度 | LeWM | 本设计 | 状态 |
|---|---|---|---|---|
| 1 | **读取** | 每帧全读、全帧全 ViT | **选择性读**：中央凹（空间稀疏）+ 偶尔睁眼（时间稀疏），靠蒸馏逼近全读 | 愿景（§7） |
| 2 | **状态** | 每帧一 token 的滑动窗 | **N 个持久残差潜向量**（渐进分解，非物体 slot），就地改写（RSSM 式信念） | 已落地（§1/§3） |
| 3 | **索引** | 严格帧序号、规则稠密 | **(时间, 空间) 标签**，稀疏不规则事件 | 部分（时间已落地：可变 Δt + 连续时间编码；空间标签随中央凹一并属愿景） |
| 4 | **输出** | 只出下一帧嵌入 | 世界信念 Z（探针可读）+ **动作潜向量→定时动作序列** + gaze/wake | 部分（Z + 动作规划已落地；gaze/wake 属愿景） |
| 5 | **可控性** | 隐式动作条件 | **显式可控闸 c**（逆动力学接地）+ **随机隐变量 ξ**（吸收不可控分量） | 已落地（§4） |
| 6 | **用途** | 给规划用的世界模型（试动作 MPC） | 快速迁移底座 + 实时指导 + 推理期反应式策略，世界模型退训练期 | 方向（§6） |

注意 SIGReg **不在**区别里——它是共享的，我们借的。区别全在"怎么读、状态长啥样、输出什么、用途"，不在防坍缩。

---

## 3. 现状：活模型闭环（Δz-JEPA Minecraft）

本节描述 [net/world_model.py](../net/world_model.py) 的 `MinecraftWorldModel` **已实现**的结构。它是 §0–§2 立场在 Minecraft 离线 VPT 录像上的已落地子集：把 §0 的"脑内世界"做成**以动作为条件、在潜空间预测增量 Δz** 的自监督世界模型。

```
img_t ─ 冻结骨干(DINOv3 ViT-S/16) ─ patch ─ proj ─ SlotBinder(固定锚) − 锚 ─→ z_obs(t)  [在线]
img_t ─ 冻结骨干(共享特征)        ─ patch ─ proj_ema ─ binder_ema − 锚 ───→ z_tg (EMA 目标, stop-grad)

   [Z=z_ref, text, h(记忆), dt, ξ, a_hist, a_cur, u(动作查询)] ─→ Transformer ×L
                                                                         │
   解码头：StateDecoder(μ=Δz 预测, c=逐 slot 可控闸, exist) · action_plan(定时动作序列)
```

要点（每条都已在活模型实现）：

- **冻结视觉骨干**：DINOv3 ViT-S/16（HF，默认；dinov2 ViT-S/14 开放权重备选），**冻结**。从零训练的随机卷积骨干没有先验、泛化失败（prime-leaf-7 实测：把纹理背熟但 holdout 退化）；冻结预训练骨干给目标编码一个独立于本任务数据的意义来源——这是 JEPA 的前提。骨干冻结且在线/目标共享 ⇒ 特征每帧只提取一次，EMA 只覆盖可训练部分（proj + binder）。
- **统一锚坐标系感知（correct）**：`proj → SlotBinder(固定锚) − 锚`。锚是与内容零互信息的常数 buffer；减锚后编码 = 纯内容增量。在线（梯度进 proj+binder）与 EMA 目标（proj_ema/binder_ema，no_grad）**同构**——同一帧两条路径得到同坐标系下同一向量（至 EMA 滞后）。时间条件不进感知输入（移到 h token），保证逆动力学差信号 `(z_tg − z_obs)` 干净。
- **预测目标 = Δz（不是绝对 z）**：旧版预测绝对潜表征时，静态内容占目标能量约 99.8%、动力学仅约 0.2%，persistence 基线不可战胜。改预测 `Δz = sg[enc(img_{t+1}) − enc(img_t)]` 后，persistence 退化为"预测 0"，动力学占目标能量 100%，动作信息成为唯一可用的预测来源——这是"训练几乎无效"根因的修复。
- **EMA 目标编码器（JEPA 稳定靶，stop-grad）**：目标走在线权重的 EMA 副本，慢速跟踪。骨干冻结 ⇒ 非平稳只可能来自 proj/binder，EMA 恰好罩住这两处；在线权重的任何坍缩动作要经 `ema_decay` 低通才会进目标，期间 SIGReg 与逆动力学有时间纠偏。**目标必须 stop-grad（I8）**，否则"减小预测误差"可靠移动靶子白嫖（让 encoder 把目标变得好预测）= 坍缩。
- **Transformer 动力学推演**：双向无掩码，一次推演全部 token。跨步记忆只走 **h token**（其余每步重新编码）。
- **StateDecoder → μ / c / exist**：μ 是 Δz 预测；c 是**逐 slot 标量可控闸**（sigmoid 有界，I3）；exist 是存在概率。末层零初始化 ⇒ 冷启动 μ=0（恰为 persistence 基线，归一化 pred 损失从 1.0 起步）、c=0.5（居中待极化）。**σ 异方差支路已撤**（见 §4）。
- **可变 Δt（jumpy prediction）**：每个转移跨度 `dt ~ U{1..max_skip}`（帧）由数据集采样。模型同时收 (a) 区间内**完整原始动作序列** `a_cur`（带有效位区分零填充与真·无操作）、(b) 聚合动作历史 `a_hist`（各条目用 `ContinuousTimeEncoding(t_hist)` 标"几帧前"）、(c) `ContinuousTimeEncoding(dt)` 条件 token。固定步长会让模型学"默认漂移先验"；Δt 可变后唯一能解释 Δz 的就是把区间内动作逐个积分——这正是开环推演所需能力，且动作效应随 Δt 累积而编码噪声地板不变，大 Δt 样本信噪比更高。
- **plan-as-vector（动作规划头）**：DETR 式 K 个动作查询，各解码出（按哪键、onset Δt、时长、是否真有），一次输出一段**定时动作序列**而非单步动作。时间锚契约：`t_vec` 是本次前向的"现在"，也是一切输出的 0 时刻（onset 从这一刻起算，单位 = 帧，与 dt 同单位）。
- **任务文本条件**：冻结句向量（[domains/minecraft/task_text.py](../domains/minecraft/task_text.py)，384 维）→ 线性投影 → text token。单任务数据下文本是常数（零互信息）；多任务混采时它解释任务间行为方差。不传时回退可学常数 placeholder（等价无条件）。
- **防坍缩两道闸**：SIGReg（训练端施加在在线 `z_obs` 上，坍缩发生处）+ `slot_diversity_loss`（§1）。

**训练时序（teacher forcing）**：每步感知输入 = 在线编码 `z_obs(t)`，μ 预测 `Δz(t→t+dt)`；跨步记忆只走 h。开环推演（可视化/诊断）用 `ẑ(t+dt) = ẑ(t) + μ(t)` 累积。

---

## 4. 可控性与不确定性

- **可控闸 c（已落地）**：每个潜向量自带一个 `c_i ∈ [0,1]`（sigmoid 有界，I3）。动作能解释某槽多少变化，它就多可控。c 由**逆动力学头**接地——[heads.py](../net/heads.py) 的 `InverseDynamicsHead` 从潜变化 `(z_tg − z_obs)·c`（槽路）+ 冻结特征的 patch 平均 Δz（旁路）+ 上下文 h 反推动作；两路**分开监督**避免 patch 旁路把槽路的梯度饿死。逆动力学只进损失、不进前向（I6 精神）。
- **随机隐变量 ξ（已落地，接替 σ）**：Δz 里"转身揭示的新内容"本质不可预测，确定性 μ 只能输出模式平均。ξ（Dreamer 式）给不可预测部分一个**有价格的去处**：
  - 后验 `q(ξ|ctx, Δz)` 训练时看真实 Δz；先验 `p(ξ|ctx)` 只看当下；`KL(q‖p)`（`β_kl`）是通道的价格。
  - `μ = f(z, a, ξ)`：闭环训练用后验采样（主干不再为不知道的事赔钱）；开环/eval 用先验**均值**（诚实口径）；想象/推演从先验采样。
  - `xi_proj` 零初始化 ⇒ 通道从静默开始，有利可图才被打开（KL 曲线 = 用量计）。后验对 Δz 取**逐槽** φ 投影后 mean+max 双池化（新内容是局部细节，跨槽平均会抵消；max 保住某槽突现的意外），带宽由 φ 维度把守。
- **为何撤 σ**：异方差 NLL 让 σ 自由时，loss 可靠"标定残差"下降（学会把梯度静音而非把误差降低），pred 曲线失真。ξ 把"不可控 = 不确定"从一个泄压阀换成一个**计价通道**：确定维要求准，不可控维有处可去且付 KL。

---

## 5. 训练设计

**全程自监督**（无人工标注）：预测 target 来自模型自己（EMA）编码的未来帧；动作 target 来自录像里**实际执行过的真动作**；不确定性由 ξ/KL 自标定。活模型的复合损失（见 [train/minecraft/losses.py](../train/minecraft/losses.py)）：

```
L = L_pred(Δz) + λ·L_sigreg + β·L_inv + β_div·L_slotdiv + L_plan_bc + β_kl·KL(ξ)
```

- **L_pred**（`dz_pred_loss`）：`μ` 逼近 **stop-grad** 的 `Δz_tg`（EMA 目标差），潜空间算（JEPA，不回像素），按 persistence=0 归一化（基线 1.0）。
- **L_sigreg**：把在线 `z_obs` 钉到各向同性高斯（Epps-Pulley sliced 检验），方向/幅度拆开接（§1）。见 [blocks/primitives.py](../blocks/primitives.py) 的 `SIGReg`、[tests/unit/test_sigreg.py](../tests/unit/test_sigreg.py)。
- **L_inv**（`minecraft_inv_dyn_loss`）：逆动力学，给可控闸 c 接地（§4）。
- **L_slotdiv**（`slot_diversity_loss`）：竞争注意力成对重叠软惩罚（§1）。
- **L_plan_bc**（`plan_bc_loss`）：动作规划头对录像真动作做行为克隆（定时序列监督）。
- **KL(ξ)**（`kl_diag_gauss`）：ξ 通道价格（§4）。

**三道防退化闸（设计原理，缺一即坍缩成 trivial 零 loss）**：

1. **SIGReg 防坍缩**——否则"潜向量坍成常数 ⇒ 预测误差恒 0 ⇒ loss=0 却什么也没学"是全局最优。弃 Mamba 递归凸更新后，此项是**承重**不是可选。
2. **target stop-grad（I8）**——否则减小预测误差可靠移动靶子白嫖（§3 EMA 段）。
3. **Δz 目标本身**——预测增量而非绝对值，堵死"重编码当前帧即得低 loss"的平凡解逃生通道（§3）。这是上一版"训练几乎无效"的真正根因修复。

**两条跨任务设计戒律（从早期实验提炼，活模型沿用）**：

- **截断 BPTT**：每步若 detach 断掉跨时间梯度，动作时序因果学不了；截断窗（k 步）让因果信号爬起。
- **多实体读出必须结构化 query、不能池化**：`mu.mean(slot)` 把单实体的动作效果稀释进 1/N → 读不出；改用 per-实体 query 对 slot 做 cross-attention。逆动力学/动作规划/探针的读出头都遵此（印证 §1 残差 slot 戒律）。
- **真动作喂 dynamics**：训练 predict 吃实际执行过的真动作，不吃模型自己预测的动作（否则模型挑"好预测的动作"自欺）。动作规划头是 Z 上的**读出头**，其梯度不在同一次前向回流去喂 dynamics。

---

## 6. 北极星：看视频掌握玩法（in-context，方向）

**当前北极星，落成一条可证伪命题**：把"用试错学会玩法"压成"看一段视频 + 试跑几步，靠**脑内记忆**在**权重冻结**下 in-context 适应新玩法"。

- **杠杆**：训练/测试**环境分布 disjoint**——测试时遇到的控制映射/任务在训练里没出现过，唯一能解释适应的就是 in-context（h 记忆）而非权重已背下答案。
- **已就位脚手架**：[domains/minecraft/control_remap.py](../domains/minecraft/control_remap.py)（逐 episode 控制重映射，零新数据即制造"同一画面、不同操作语义"的 in-context 信号）；[task_text.py](../domains/minecraft/task_text.py)（任务条件）。
- **证据档**：① 逐 episode 操作重映射（看一段被重映射的视频 → 推出本 episode 的操作语义）；② episodic loss-on-query（在 query 段上算适应增益）。
- **协议**：baselines（无记忆 / 随机 h）对照 + dose-response（适应增益随 context 视频长度单调）。真 Minecraft 活环境（craftground，headless 软渲染）解锁主动交互轴 + rollout 通关率，离线数据测不了这些。
- **已知架构缺口**（待补，非阻塞）：逆动力学头对 context 全盲（看不到适应信号）；单个 h token 太细，承载不下一整套玩法记忆；缺"有用 context"的梯度压力（模型没有动机真去用 context）。这些是把方向变成结果前要堵的洞。

---

## 7. 更远的愿景：选择性读取（fovea / gaze / wake，无代码）

⚠️ 本节描述**尚未落地**的设计愿景，**当前仓库无对应代码**（原 `tao_not_42.py` 的 fovea/gaze 实现已删）。保留它是因为：当"读取成本"成为瓶颈时，这是既定演进方向。

- **双通道感知 = 余光 + 中央凹**：余光廉价常开、给全局显著性；中央凹只对注视点的局部裁剪做高清 ViT。这是"偶尔扫一眼"的物理实现——不把整帧高清灌进脑子。
- **主动感知出口 = gaze**：决定下一眼看哪 `(g_x, g_y, g_s)`，经可导的 STN（affine_grid + grid_sample，fp32 满足 I4）裁剪——**"看哪里"可被梯度直接学**，不必退回高方差硬注意力 RL。空间标签用 `SpatialPosEmbed`（已在 [blocks/primitives.py](../blocks/primitives.py)，待接入）。
- **异步节律 = wake**：输出"多久后再睁眼"，配合连续时间编码兜底稀疏/不规则观测。
- **训练靠两阶段自监督蒸馏（绕开 RL）**：阶段一全观测训出动力学与表征（无鸡生蛋）；阶段二冻成老师，gaze/wake 用**从自身预测误差算出的 oracle 当自监督标签**（误差热图监督看哪里、想象 rollout 首次发散时刻监督何时看）。**观测必须有代价** `cost(obs)`——否则"最大降低预测误差"的最优解是全看，退回稠密感知；这一项才逼出稀疏读取。

---

## 8. 数值不变量映射（I1–I8）

| 机制（活模型） | 满足 |
|---|---|
| `SlotBinder` 门控残差注入、各更新增益受限 | I5（增益受限、非扩张） |
| StateDecoder 的 c/exist 走 sigmoid、ξ logvar clamp、动作头有界 | I2 / I3（有界、log 安全） |
| 危险算子（normalize/除/exp）强制 fp32 | I4 |
| 感知/推演全程 LayerNorm，无 BatchNorm；Transformer dropout=0 | I7（递归路径稳；train/eval 前向一致） |
| EMA 目标 stop-grad、不取消 detach | I8 |
| SIGReg、逆动力学、Sinkhorn 类统计项只进 loss、不进前向 | I6 |

---

## 9. 证伪探针与诊断

- **薄探针原理**：检验"Z 是否真存了世界"的探针须**刻意保持极薄**（单隐层），且从**聚合后的全体残差潜向量**读出、非某个固定 slot——读得准只能归功于 Z 本身确实存了世界，而非探针脑补。[net/world_probe.py](../net/world_probe.py) 的 `WorldProbeDecoder` 是这类探针的实现（独立模块，当前未接入 Minecraft 主循环）。
- **活诊断（已接入）**：
  - [tools/oracle_idm.py](../tools/oracle_idm.py)：逆动力学上界——冻结特征里到底能读出多少动作，钉死瓶颈在编码还是读出。
  - [train/minecraft/eval.py](../train/minecraft/eval.py) 的 `rollout_probe`：开环累积 `ẑ+μ` 与真相的发散曲线，量化前向模拟能撑多久。
- **指标戒律**：① 低 loss ≠ 世界模型，须用抗作弊指标（开环对照，骗不了位置先验）；② 多实体读出结构化 query、不池化（§5）；③ 运动样本要单独看（全步 RMS 会被静止帧稀释）。

---

## 10. 待建缺口（live，不阻塞）

1. **in-context 北极星的架构补洞**（§6）：逆动力学头接 context、h 记忆扩容、给"用 context"加梯度压力。
2. **真环境闭环**：craftground 活环境对齐 22 维键契约，跑 dose-response 与 rollout 通关率（离线数据测不了主动交互）。
3. **选择性读取**（§7）：中央凹/注视/唤醒 + 观测代价 + 两阶段蒸馏，全部待建（无代码）。
4. **空间标签接入**：`SpatialPosEmbed` 已在 blocks，待随中央凹接入感知端。
5. **残差结构防坍缩诊断**：per-slot 方差谱（应递减）+ 跨 slot 相关（应低）作为常设面板。
6. **VPT teacher 蒸馏**（[train/vpt/distill_vpt.py](../train/vpt/distill_vpt.py)）：软 KL 把 OpenAI VPT 策略边缘化到 22 维契约，给动作头一个强先验。
