# 结论(中期):R2a 采样发散度探针首个数据点(2026-07-03)

> 对应 [design_rollout_research_program.md] §R2a。探针脚本 `tests/probe_r2a_divergence.py`,
> 训练管线 `train/crafter/train_dreamerv3.py --size small`(6.7M 参数,200k 步,runs/crafter_r2a)。
> 本文档记录 4k 步早期检查点的冒烟数据点;**闸门裁决须等 60k+ 成熟检查点复测,此处不下最终结论**。

## 执行记录

- 11:12 UTC:g500_dino 解码头训练在 14,700/20,000 步终止(用户授权;holdout 峰值
  22.32dB@7k 已由 best.pt 保全,后段为解码头过拟合,见 [conclusion_g500_gates_probe.md] 线索)。
  看门狗随即自动拉起 Crafter DreamerV3 训练。
- 11:17 UTC:首个 checkpoint(4k 步)出现,探针冒烟通过(修复 `VecCrafterEnv.step`
  五元组解包后),产出首个数据点 `runs/crafter_r2a/probe_divergence.json`。
- 12:40 UTC 状态:52,000/200,000 步(26%),9 sps,ep_rew 2.54,ach/ep 3.44,
  max_score 7.1,train 覆盖 11/22 成就(新增 collect_stone / make_stone_sword /
  make_wood_pickaxe),WM loss 45→39 缓降。预计还需 ~4.5h 跑完。
- 已布防:60k checkpoint 出现时自动跑正式规模复测
  (`--collect-steps 512 --n-envs 8` → probe_divergence_60k.json)。

## 首个数据点(ckpt 4,000 步,109 窗口,K=15,N=8)

| k | 1 | 2 | 3 | 4 | 5 | 6..15 |
|---|---|---|---|---|---|---|
| Spearman ρ | **0.64** | **0.58** | **0.38** | -0.03 | -0.20 | 全部 ≈0 或弱负 |

- mean per-k ρ = **-0.017**(判据口径:≥0.5 PASS / <0.3 FAIL)→ 按 4k 点算 FAIL,
  但这是**未成熟 WM 上的预期形态**,不作裁决。
- pooled ρ = -0.098:与 per-k 均值同号且更低,证实了设计时的担忧——pooled 指标
  会被 k 趋势污染,逐 k 相关必须是主指标(本探针方法论上站住了)。

## 解读与可证伪预测

短视野(k≤3)相关性显著为正:发散度在模型尚有信号的深度内确实追踪真实开环误差。
k≥4 断崖归零:4k 步的 RSSM 想象在 4 步外已是噪声,发散度与误差都饱和,排序信息消失。

**可证伪预测**:若发散度机制成立,随 WM 成熟(60k / 200k 检查点),正相关的有效 k 区间
应向右扩张(4k 时 k≤3 → 60k 时显著更深),mean per-k ρ 相应抬升。
- 若 60k 复测 mean ρ ≥ 0.5 → R2a PASS,发散度采纳为想象截断/降权信号,开工 R3;
- 若有效区间不扩张(仍卡 k≤3)→ 机制本身与 WM 成熟度无关,FAIL 坐实,回退 K 头集成;
- 中间态(0.3–0.5)→ 等 200k 终点检查点再测一次。

## 操作性副产物

- `VecCrafterEnv.step` 返回五元组 `(obs, rewards, dones, infos, new_achievements)`,
  下游消费方注意解包(探针已修,commit 4406087)。
- 探针成本极低(单检查点 <2min L4),可作为训练常态伴随诊断按 checkpoint 周期跑,
  得到"有效想象深度 vs 训练步数"曲线——这条曲线本身就是 R3 自适应视野的先验输入。
