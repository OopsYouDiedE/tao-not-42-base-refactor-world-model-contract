# 世界模型版图调研与经验提炼

> **文档性质**:对近年世界模型 / 像素生成世界模型 / 游戏通用操作模型的外部调研,以及对本项目设计选择的对照与经验。
> 不描述本仓库的微观实现(归代码与 [mental_world.md](mental_world.md));只做外部参照、设计原理对比与可选方向记录。
> 本项目定位见 [mental_world.md](mental_world.md):Δz-JEPA、冻结 DINOv3、潜空间预测 Δz、EMA 目标、逆动力学接地可控闸 `c`、有限持久潜槽、Transformer。

---

## 1. 两条正交轴:别把"生成"与"自回归"混为一谈

近年模型用两条独立的轴定位即可避免概念混淆:

- **轴 A — 输出空间**:像素 / token 生成(解码回画面) vs 潜空间预测(只预测表征,不解码)。
- **轴 B — 时间推进**:自回归 rollout(把预测喂回作为下一步输入,多步想象) vs 单步 / 能量规划。

**关键澄清**:本项目"不做自回归"针对的是**轴 A 的像素自回归生成**。而**轴 B 的潜空间 rollout** 是另一回事——DINO-WM、V-JEPA 2-AC、DreamerV3 都在**潜空间里自回归 rollout** 做规划,且都不解码像素。因此本项目的立场应精确表述为:**不做自回归像素 / token 生成;潜空间多步想象不在禁止之列**,它与 Δz 预测、变 Δt 兼容,且是潜空间规划的前提。

| | 轴 A: 像素生成 | 轴 A: 潜空间预测 |
|---|---|---|
| **轴 B: 自回归 rollout** | Oasis / MineWorld / GameNGen / Cosmos / Genie | DINO-WM / V-JEPA 2-AC / DreamerV3(潜空间想象) |
| **轴 B: 单步 / 能量规划** | —— | 本项目当前的 Δz 单步预测(+ 训练期可扩展为想象) |

---

## 2. 与本项目最近的开源模型(代码 + 权重可得)

这三者与本项目同属"轴 A 潜空间预测",设计原理最值得对照。

### 2.1 DINO-WM —— 几乎是本项目的同构体

- 来源:arXiv 2411.04983,ICML 2025(NYU / Meta)。
- 设计:冻结 **DINOv2 patch 特征** + ViT 转移模型,预测**未来 patch 特征**,**无像素解码器**(可选解码器仅供可视化)。
- 规划:离线轨迹自监督训练后,推理期用 **MPC(CEM / MPPI)在潜空间最小化"预测潜 → 目标 patch 特征"的 MSE**,把目标特征当预测靶。
- 结果:零样本在迷宫 / 推物 / 多粒子等 6 个环境完成目标到达,报告优于 DreamerV3 与 IRIS;**不需要预学逆动力学模型**即可规划。
- **对本项目**:证实"冻结 DINO + ViT 潜预测 + decoder-free"是可行主线。差异有二:① 我们把 patch 进一步压到 **N 个实体槽**——更激进的压缩,换来实体级可控,但有信息瓶颈风险(对应 oracle_idm 触顶的历史观察);② 我们额外有 `c` 闸 + 逆动力学接地(`net/world_model.py` 的 `inv_dyn` 与 `train/minecraft/losses.py:minecraft_inv_dyn_loss`),可视作 DINO-WM 的超集。**最大差异:DINO-WM 在推理期跑 MPC,我们当前不跑(见 §5)。**

### 2.2 V-JEPA 2 / V-JEPA 2-AC

- 来源:arXiv 2506.09985,Meta,2025-06;代码与权重已开放。
- 设计:1.2B 编码器,**action-free** JEPA 在 >100 万小时视频上预训练;预测在表征空间,**EMA 目标 + stop-grad** 防坍缩(JEPA 标志做法)。
- **V-JEPA 2-AC**:300M、24 层 transformer 的动作条件预测器,仅用 **<62 小时** Droid 机器人数据后训练;**冻结编码器,只训预测器**。
- 规划:**想象候选动作 → 按潜空间到目标距离打分 → MPC 重规划**,无像素解码、无奖励;Franka 抓放零样本成功率 65–80%。
- **对本项目**:① 两阶段范式(海量 action-free 表征预训练 → 小样本动作条件后训练)正映射本项目"看视频 → in-context 适应";② "冻结大表征 + 训练薄动作预测器"的清晰切分,可对照本项目 `inv_dyn_ctx`(FiLM-on-h,`net/world_model.py`);③ EMA + stop-grad 双保险,与本项目 EMA 目标编码器(`encode_target`)方向一致。

### 2.3 DreamerV3 —— 鲁棒性配方可直接借

- 来源:arXiv 2301.04104,Nature 2025;`github.com/danijar/dreamerv3` 开源。
- 设计:**RSSM = 确定性递归状态 h + 随机离散 latent**,**含解码器(重构式)**;在潜空间想象 rollout 上做 actor-critic;首个从零(无人类数据 / 课程)拿到 Minecraft 钻石。
- 鲁棒性配方:symlog、two-hot 回报、**free-bits KL**、跨 150+ 任务**单一超参**。变体 MuDreamer 去掉重构。
- **对本项目**:① RSSM 的"确定性 h + 随机 latent"对应本项目 `h + ξ`(`net/world_model.py` 的 `d_xi` 隐变量);② **free-bits KL 可直接用到 `train/minecraft/losses.py:kl_diag_gauss` 防后验坍塌**;③ Dreamer 经验显示**离散 latent 在视觉游戏更稳**,可评估把 ξ 的连续高斯补 / 换成离散类别;④ symlog / two-hot 适用于有界回归。

---

## 3. 像素生成交互世界模型(对照类,开源,反面教训)

这一类属"轴 A 像素生成 + 轴 B 自回归",与本项目路线相反,主要价值是反面教训。

- **Oasis**(Decart + Etched, 2024-11):扩散 transformer,**逐帧自回归**,键鼠条件,360×360 @ 20fps,**500M 开放权重 + 推理码**。长程**漂移**明显。
- **MineWorld**(Microsoft, arXiv 2504.08388, 2025):视觉-动作**自回归 transformer**,图像 + 动作 tokenizer 交错,next-token 训练,并行解码加速到 4–7fps,**开源代码 + 权重**,报告优于开源扩散世界模型。
- **GameNGen**(Google, 2024):扩散,神经版 DOOM,实时自回归。
- **Cosmos**(NVIDIA, arXiv 2501.03575):世界基础模型平台,tokenizer(连续 + 离散) + 扩散与自回归两族,4–14B,**开放权重**;定位"基础 WFM → 下游 fine-tune"。
- **Genie 3**(DeepMind, 2025-08):实时 24fps / 720p,分钟级一致性;Genie 2 = 自回归潜扩散(因果 transformer 预测下一潜 + 扩散头渲染)。**非开放权重**。

**反面教训**:这些模型证明 Minecraft 类动力学从 VPT 类数据**可学**,但暴露**漂移 / 长程记忆丢失 / "看着对 ≠ 是对"**(像素相似不等于状态正确)。这正是本项目选潜空间、选**有限持久潜槽**(结构性的长程记忆,对冲漂移)、选 JEPA(不在无关视觉细节上耗费容量)的理由。经验:**评测口径坚持潜空间预测度量**(pred_move / persistence 基线,见 `train/minecraft/eval.py`)而非视觉相似度;不被像素 demo 的观感带偏。

---

## 4. 潜动作模型(对北极星"看视频掌握玩法"最相关)

- **Genie 潜动作 / LAPO(arXiv 2410.11758)/ VideoWorld / UniVLA / Co-Evolving LAWM**:用 **IDM + FDM + VQ 瓶颈**从**无标注视频**推断离散"潜动作码",再用少量标注接地到真实控制。
- **对本项目**:本项目已有真实 22 维动作契约 + 逆动力学头,但北极星的"换一套玩法"目前靠**手写** `domains/minecraft/control_remap.py`。潜动作文献提示一条升级路线:把它从"手指定"改为"**每 episode 可推断的潜动作码**"——在 episode 维加一个被 episodic loss 推动的潜变量,给 in-context 适应一个"可学的钩子",对应 [mental_world.md](mental_world.md) 北极星章节点出的架构缺口(inv-dyn 对 context 全盲 / 单 h 太细 / 缺 use-context 梯度压力)。

---

## 5. 推理期速度与 amortization:本项目"退到训练期"是对解

一个常见追问:"既然 DINO-WM / V-JEPA 2-AC / DreamerV3 的核心收益都在推理期规划,它们怎么做到足够快的反应速度?" 调研结论:**全 online MPC 探索本质上慢,真正的速度来自 amortization(把探索蒸到反应网络)**。三段谱:

1. **它们大多不需要快**:V-JEPA 2-AC 用 CEM 800 样本、**约 16 秒 / 动作**(比 Cosmos 的 4 分钟 / 动作快约 16×)。机器人抓放容忍秒级控制,**不是 20fps 的反应频率**。
2. **DreamerV3 推理期不做探索**:想象 rollout 在**训练期**完成,actor-critic 在想象轨迹上学好;部署时 actor 是**单次 forward** → 实时。**探索被 amortize 进反应策略。** 这是高 Hz 控制的主流解。
3. **online 也要压成本**:潜空间(无像素解码) + 短 receding horizon + **每 K 帧 replan、间隔 open-loop** + GPU 并行候选评估;2026 出现的 "Amortizing Planning in World Models" 方向把探索蒸馏成快网络。

**对本项目的印证**:readme 的"世界模型退到训练期、推理期不在控制环跑 rollout"**正是 DreamerV3 的选择**(实时制约下的对解);本项目 plan-as-vector 行动头读 `h` 本身就是 amortized 反应策略,已在实时友好的一端,无需在"慢 MPC vs 无规划"之间二选。北极星同样不要求推理期探索:**"看"= 更新持久潜 `h`(廉价 forward),"动"= 反应头读 `h`(廉价 forward)**,in-context = **h 条件化**而非 MPC,高成本付在训练期。

**措辞建议**(供 mental_world / readme 后续修订):"退到训练期" = *不在控制环跑 online 探索*,**并非** *永不做多步潜想象*。`train/minecraft/eval.py:rollout_probe` 的演化方向是**训练期 imagination 信号**,不是推理期控制器;若 `h` 条件化不足,逃生口是**每 K 帧的有界 online refinement**(receding horizon 保持廉价),而非全程 MPC。

---

## 6. 经验对照表

| # | 经验 | 来源 | 对本项目的落点 |
|---|---|---|---|
| 1 | 冻结视觉特征 + ViT 潜预测 + decoder-free 是可行主线 | DINO-WM | 增强核心赌注信心;本项目是其超集(多 `c` 闸 + 槽化) |
| 2 | 世界模型可用于推理期潜空间规划,但成本高 | DINO-WM / V-JEPA 2-AC | 见 §5:本项目"退到训练期"是实时对解,非缺陷 |
| 3 | "自回归"分像素与潜空间两义;潜 rollout 必需且兼容 | 全体 | §1 的轴 A / 轴 B 区分,建议写进 mental_world / readme |
| 4 | 两阶段:海量 action-free 预训练 → 小样本动作条件后训练 | V-JEPA 2 / Cosmos | "冻结大表征 + 薄动作预测器"切分,对照 `inv_dyn_ctx` |
| 5 | 潜动作码可从无标注视频推断(IDM + FDM + VQ) | Genie / LAPO | `control_remap` → 每 episode 可学潜动作码 |
| 6 | free-bits KL 防后验坍塌;离散 latent 在视觉游戏更稳 | DreamerV3 | `kl_diag_gauss` 加 free-bits;评估 ξ 离散化 |
| 7 | 像素模型漂移 / 长程记忆丢失 → 潜 + 持久槽是结构性答案 | Oasis / MineWorld | 坚持潜空间评测口径;持久槽对照像素漂移论证 |
| 8 | anti-collapse 是潜世界模型硬约束(EMA + stop-grad + 方差正则) | V-JEPA 2 / LeWorldModel | 维持 `SIGReg`(`blocks/primitives.py`) + `slot_diversity_loss`;对照检查 EMA 目标稳定性 |

---

## 7. 后续候选项(未实施,待评估)

以下为调研引出的可选改进,本轮不实施,留待后续指令逐项评估:

1. **`kl_diag_gauss` 加 free-bits 下限**(Dreamer 配方):防 ξ 后验坍塌。低风险,与现路线兼容。
2. **评估 ξ 离散化**:连续高斯 → 离散类别 latent,参 Dreamer 在视觉游戏的稳定性观察。
3. **`control_remap` 可学化**:手写重映射 → episode 级可推断潜动作码,受 episodic loss 驱动(§4)。
4. **`rollout_probe` → 训练期 imagination loss**:把离线诊断升级为训练期多步想象监督(§5),而非推理期控制器。
5. **轴 A / 轴 B 区分写入 mental_world / readme**:澄清"非自回归"的精确含义(§1)。

---

## 8. 来源

- V-JEPA 2:arXiv 2506.09985;Meta 博客 `ai.meta.com/blog/v-jepa-2-world-model-benchmarks/`;16s/动作 数据 `gonzoml.substack.com/p/v-jepa-2-scaling-v-jepa`
- DINO-WM:arXiv 2411.04983;项目页 `dino-wm.github.io`
- DreamerV3:arXiv 2301.04104;Nature 2025 `s41586-025-08744-2`;代码 `github.com/danijar/dreamerv3`
- Oasis:`oasis-model.github.io` · MineWorld:arXiv 2504.08388;`github.com/microsoft/mineworld`
- Cosmos:arXiv 2501.03575;`github.com/nvidia-cosmos/cosmos-predict1`
- Genie 3:`deepmind.google/blog/genie-3-a-new-frontier-for-world-models/`
- 潜动作:LAPO arXiv 2410.11758;世界模型 2025 阅读清单 `medium.com/@graison`
- 速度 / amortization:Amortizing Planning arXiv 2605.08732;FF-JEPA arXiv 2606.09311
- anti-collapse 血缘:LeWorldModel arXiv 2603.19312
