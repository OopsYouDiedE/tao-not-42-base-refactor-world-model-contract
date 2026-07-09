---
name: lessons-do-not-retry
description: 别再试登记表——只收"方案本身不成立"的负结果(实验做对、口径没错、结论就是走不通)。自己的 bug/错误理解/被推翻的归因一律不收。一行一条,供未来提案前检索。
metadata:
  type: knowledge
---

# 别再试登记表

收录判据:只登记**方案本身不成立**的负结果——实验做对了、口径没错、无 bug,跑出来
就是"这条路走不通"。自己的 bug、当时的错误理解、被后续推翻的归因,一律不收。
格式:`别再做 X — 因为 Y(关键数字) — 证据:<file:line 或 runs/*.json>`。

## 训练信号与评价

- 别再用手工统计量(命中率/进度分)作训练信号或主指标 — 实测恒定复读臂与被测臂同分
  (0.75 vs 0.75)、随机臂反而更高(0.438);里程碑只作不可刷的汇报锚点,不进训练信号 —
  证据:docs/activity_log.md:379-394。
- 别再把 LLM 判官的严格排序当"学到东西"的证据 — 判官对纯噪声(无意义色块)也编造语义并给
  全序,故 adv_var>0 不构成证据 — 证据:docs/activity_log.md:404-406(probe_judge_io_haiku)。
- 别再用离线 acc / BC loss 当闭环行为有效性的证据 — D 曲线:同配方同规模跨示范子集,离线
  acc 几乎相同而闭环 arrive 在 0.06–0.81 乱跳 — 证据:docs/architectures/fovea-experiments-index.md:260-274。
- 别再在高帧率下用 raw PSNR 对比 persistence 判世界模型 — 指标结构性偏袒复读(重建税+静止
  像素灌水),是数学性质非模型无能;用 EV(Δz)/开环 — 证据:knowledge/conclusion_minecraft_dreamer4_run.md:157-170。

## 教师与可学性

- 别再 BC 一个不是学生观测函数的教师(raycast 特权),或动作含不可观测潜变量的教师(随机搜索
  方向)— 前者视觉学生天花板锁死,后者同观测标签 ±15° 对冲、BC 均值归零、切换率钉死 0.17 —
  证据:docs/architectures/fovea-brain-division-scale-plan.md:117-120,149-151;fovea-experiments-index.md:42。
- 别再指望慢塔 Mamba belief 比单帧向量更能减少学生要学的 — ΔR²=−0.02(线性);**限定**:该实验
  教师是 raycast 特权驱动、动作非视觉函数,结论受混淆非定论 — 证据:knowledge/conclusion_fovea_ceiling_mamba_seed.md:36-42。
- 别再指望预训练 LLM 的递归 SSM 状态兼职视觉世界记忆 — W4/W4b/W4c 三连败,LoRA+3k 步内解冻
  递归张量也未恢复视觉 age(边界:全参/更长预算未测);λ 谱不缺慢通道,病灶是头级全职承诺 —
  证据:docs/architectures/fovea-brain-division-scale-plan.md:31,35-43。

## 感知先验与表征

- 别再手标领域感知先验(4 类词表 / 手标分割头 / 凸包树干 GT)去救闭环 — 4 类词表把 YOLOE 每帧
  48.8 个提案、90.8% 像素覆盖压到 0.5 个框、1.3% 覆盖、准星覆盖 0%,是表征瓶颈 —
  证据:docs/activity_log.md:459-472(probe_yoloe_coverage)。
- 别再用池化+PCA 的降维代理判断表征里有没有某信息 — 方向信息存在于完整 patch 网格的空间结构,
  被 4×4 池化+PCA-256 压掉(ridge R²≈0,而 patch 互相关与 dx 相关 0.27–0.45)—
  证据:knowledge/conclusion_g500_gates_probe.md:8-18。
- 别再用原始深度图判负空间地形(洞/沟) — 掠射角下深度与远壁近乎连续,hole mIoU 0.156;正确形式=
  反投影高度图(同点 0.807)— 证据:docs/architectures/fovea-experiments-index.md:146-160。
- 别再对推理做 2× 分辨率放大以求更细感知 — 纹理尺度偏移,mIoU 0.530→0.248 —
  证据:docs/architectures/fovea-experiments-index.md:29。

## 动作头与精修

- 别再用分块动作头做反应式策略(chunk_k>1)— 离线首步 acc 随 k 单调降、闭环 switch 随 k 崩 —
  证据:net/pixel_tower.py:45(R-B 裁决)。
- 别再指望视觉平均 BC 学会"精准+稀疏+零容差+帧不可分辨"的动作(GUI 合成)— 教师 100%、BC holdout
  loss 0.0013、闭环 0.00(学到背景,漏掉稀疏关键帧)— 证据:knowledge/conclusion_fasttower_skill_ceiling.md:61-67。
- 别再按自身误差重加权(OHEM / hard_weight=1.0)— 破坏流匹配收敛,开环 −4.10dB;无帽的样本/损失
  重加权会过冲(Y2f 洞深权重峰值 w≈10,w=25 反降 0.401→0.326)—
  证据:knowledge/conclusion_minecraft_dreamer4_run.md:203-211;fovea-experiments-index.md:173。
- 别再指望 GRPO 在视觉不可见的目标上起效 — 石头贴石头墙 0.31→0.06(退化),可见的木头 0.50→0.81;
  起效前提=目标可见 + BC 暖启动 — 证据:knowledge/conclusion_fasttower_skill_ceiling.md:33-40。
