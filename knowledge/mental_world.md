# 脑内世界（Mental World）设计

> **文档定位（SSOT）**：本文只讲**宏观架构、算法意图、训练目标的设计动机与物理直觉**。
> 易变的实现细节（精确 Shape/Dtype、层数、超参）归代码本身：
> 活模型实现见 [net/world_model.py](../net/world_model.py)（部件 [slots.py](../net/slots.py) /
> [backbone.py](../net/backbone.py) / [heads.py](../net/heads.py)），积木见 [blocks/](../blocks/)，
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

## 3. 现状：活模型（序列对齐的后果结构世界模型）

本节描述 [net/world_model.py](../net/world_model.py) 的 `MinecraftWorldModel` **已实现**的结构,是 §0–§2
立场在 Minecraft 离线录像上的落地。相对早期"两步式离散动作词表 + 单步 Δz 重建"(已弃),本版把对齐
从"逐帧/对动作词表"改为**编码空间的序列↔序列对齐**:用"初始帧编码 + 一段(图像+动作)token"预测
**同一个未来帧的潜向量**,要求不同长度上下文对该未来帧的预测互相一致。设计四原则:① 对齐在编码空间
且序列↔序列;② 后果 = 下游潜发散(非瞬时像素差);③ 主动防编码偷懒;④ 可逆/不可逆因子化。

```
img_t ─ 冻结 DINO ─ patch ─ adapter ─→ z=(z_rev 有界, z_inv 随机潜+KL)  [online]
img_t ─ 冻结 DINO ─ patch ─ adapter_ema ───────────────────────────→ z̄ (EMA 目标, stop-grad)

   时空 token 集合 𝒯 = {u_{t,p}=W·z+ρ(屏幕坐标 sine2d)+τ(帧时间)} ∪ {g_n=W·a_n+τ(t_n)}
   上下文 𝒞_k(观测帧 0..k + 全程动作)  ⊕  未来 query q=m+ρ+τ(t*)  ─→ Transformer 核 ─→ ẑ_{t*}
   因子化:ẑ_rev=z_rev+Σ_j c_j·G_j(z_rev)  ·  ẑ_inv=z_inv+𝒟.decode(event)  ·  do(null) 旁路→ e
```

### 数学(8 式,代码即此)

记上下文视图 `𝒞_k`、未来 query `q_{t*,p}=m+ρ(row,col)+τ(t*)`,预测器 `P_ψ`(Transformer 核):
1. **时空 token**:`u_{t,p}=W_in z_{t,p}+ρ(row_p,col_p)+τ(t)`,动作 `g_n=W_a a_n+τ(t_n)` ⇒ 摊平成集合单次处理(省算力)。
2. **序列→未来帧**:`ẑ_{t*,p}=P_ψ(𝒞_k; q_{t*,p})`,拆 `(ẑ_rev,ẑ_inv)`。目标是任意未来 t*、由动作序列积分,**非逐帧下一帧**。
3. **EMA 教师**:`z̄_{t*,p}=sg·E_θ̄(f_{t*,p})`(I8)。
4. **序列↔序列对齐**:`L_align = Σ_k (1/M)Σ_p w·‖ẑ^{(k)}−z̄‖² + λ_agree·Σ_{k<k'}‖ẑ^{(k)}−sg ẑ^{(k')}‖²`
   ——多上下文预测同一 t* 且互相一致。
5. **后果权重**:`w ∝ norm(‖ẑ_inv(do a)−ẑ_inv(do null)‖ + α·Φ_future)`,随未来效应不随像素 ⇒ 小像素高后果不被淹没。
6. **路径无关**:`L_path=‖ẑ_inv(A)−ẑ_inv(A')‖²`(A,A' 到同一未来态)⇒ z_inv = 对可逆/路径冗余取商的"后果"。
7. **因子化动力学**:可逆 `ẑ_rev=z_rev_0+Σc_j G_j`(生成元复合);不可逆 `ẑ_inv=z_inv_0+Σ𝒟(event)`,`event=VQ(Δz_inv)`(无序累加 ⇒ 离散通道天然路径无关)。
8. **反捷径**:stop-grad 目标 + SIGReg + w 随未来发散 + 多上下文一致 + 验收 `corr(w,像素)≈0, corr(w,未来发散)>0`。

要点(均已落地):
- **冻结视觉骨干 + 可训练 adapter**:DINOv3 ViT-S/16(dinov2 备选),冻结;之上 `_Adapter`(`Linear`→PreLNAttn 自注意 + GatedResidual MLP)给编码侧容量。`unfreeze_backbone_layers>0` 时探针失败可解冻顶层(默认关)。
- **因子化潜 z=(z_rev,z_inv)**:`z_rev` 经 `BoundedActivation('flow')` 有界(I3);`z_inv` 由 `StochLatent` 给随机潜 + `β_kl·KL` 信息瓶颈(逼其只留会改变未来的位;β_kl 不能太大以免丢掉小而关键的位,由后果监督托底)。
- **EMA 教师 + stop-grad(I8)**:目标走 `adapter` 的 EMA 副本 `target_adapter`,慢速跟踪;减小预测误差不能靠移动靶子白嫖。
- **效应词表 𝔤⊕𝒟 全部从潜变化读出**:`EffectTokenizer` 对 `Δz_inv` 量化得事件码(token=不可逆后果,**非动作**);`GeneratorBank` 是可逆生成元算子组;动作只作条件 token。
- **后果加权 + 反捷径验收**:importance 由反事实效应 ‖e‖ 给出;eval 报 `corr(w,future)`/`corr(w,pixel)` 与闭环漂移、z_inv 线性探针(解 has_item/airborne)。
- **可变 Δt(jumpy prediction)**:转移跨度 `dt~U{1..max_skip}`(帧),帧时间由 dt 累计、喂帧不喂秒;未来 t* 须把区间动作积分才能预测。

**训练采样**:每个 clip 取锚点 0、目标 t*=末帧、若干上下文截止 k;构造 token 集合一次前向算对齐 + 一致损失,EMA 每优化步更新。详见 [train/minecraft/train_minecraft.py](../train/minecraft/train_minecraft.py) 的 `run_sequence`。

---

## 4. 可控性与不确定性 (后果权重与因子化潜)

- **后果权重 w (已落地)**：利用反事实效应 $\|e\| = \|\hat{z}_{inv}(do\ a) - \hat{z}_{inv}(do\ null)\|$ 衡量一个动作对世界产生的影响程度。空中跳跃或无效按键将产生 $\|e\| \approx 0$，而合成、破坏方块等将产生较大的 $\|e\|$。以此计算后果权重 $w$ 进行重加权，只参与损失加权不反传梯度 (detach)，强迫模型聚焦于低频但高影响（低像素能量、高后果）的稀有模式，解决小像素高后果被像素 MSE 淹没的问题。
- **因子化潜 z=(z_rev, z_inv) 与 KL 信息瓶颈 (已落地)**：潜空间被因子化为可逆通道 $z_{rev}$ 与不可逆通道 $z_{inv}$。
  - $z_{rev}$ 通过 BoundedActivation 和可逆生成元算子组限制过度发散；
  - $z_{inv}$ 通过 `StochLatent` 模块注入 $\beta_{kl} \cdot KL$ 散度信息瓶颈，强迫模型扔掉不改变未来的冗余信息，只保留具有不可逆后果的信息。
- **认知 surprise (已落地)**：通过 K 个轻量预测头预测未来的头间方差作为不确定性度量 (surprise)，用作未来事件分段和新奇度评估，不影响预测主干的前向计算。

---

## 5. 训练设计

**全程自监督**（无人工标注）：预测 target 来自慢速跟随的 EMA 教师目标编码器 `target_adapter` 的输出；通过可变跨度 $\Delta t$ 的 jumpy 采样以及多上下文截止集合，在序列空间中一次性前向构建对齐和一致性约束。

复合损失函数如下：
- **L_align** (`latent_align_loss`)：后果加权的潜空间 MSE 对齐损失，预测 $\hat{z}$ 逼近 stop-grad 的 EMA 教师目标 $z_{tgt}$。
- **L_agree** (`agreement_loss`)：多上下文一致性损失。不同上下文截止对同一个未来帧 $t^*$ 的预测特征，需要相互接近，在没有显式教师时强制动力学表征保持一致。
- **L_event** (`event_ce`)：辅助离散事件通道交叉熵损失。预测的事件码概率逼近由实测 $\Delta z_{inv}$ 通过 `EffectTokenizer` 量化得到的离散事件索引。
- **L_noop** (`noop_loss`)：反事实效应幅度头回归损失，预测 $\|e\|$ 逼近 stop-grad 后的实测效应幅度。
- **L_path** (`path_invariance_loss`)：路径无关损失。在同一个 clip 里，经历不同动作路径到达同一个终点状态的两个序列预测，其 $z_{inv}$ 必须重合，强制潜空间的后果对齐。
- **L_sigreg** (`SIGReg`)：Sliced 各向同性高斯正则，对在线编码 $z_{obs}$ 施加，防止表征坍缩。

**三道防退化闸**：
1. **SIGReg 防表征坍缩**：防止 $z$ 变为常数，是 JEPA 架构的核心保护。
2. **教师 stop-grad (I8)**：防止预测器和编码器协同把靶子拉向预测值，导致平凡收敛。
3. **EMA 目标慢速跟随**：教师权重以大动量（如 0.996）还原滞后滑动更新，保证目标的稳定性。

**两条跨任务设计戒律**：
- **可变时间步跳跃采样**：转移跨度 $\dt \sim U\{1..max\_skip\}$ 帧，不预测中间冗余状态，只由累积动作与跨度时间编码直接对齐到未来。
- **真动作喂 dynamics**：动作始终作为条件 token 输入参与前向预测，不作为自监督对齐的回归目标，动作分词器仅作用于脑内不可逆潜空间 $\Delta z_{inv}$。

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
- **主动感知出口 = gaze**：决定下一眼看哪 `(g_x, g_y, g_s)`，经可导的 STN（affine_grid + grid_sample，fp32 满足 I4）裁剪——**"看哪里"可被梯度直接学**，不必退回高方差硬注意力 RL。空间标签用 `SpatialPosEmbed`（已在 [blocks/encodings.py](../blocks/encodings.py)，待接入）。
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
