# Minecraft Dreamer4 离线 + 在线训练结论（2026-07-02，Colab L4）

> **第三轮（同日，e8c8904 因果修复后变体对比）已完成**，见 §00；第二轮见 §0；§1-§5 为首轮（26M/fp32）记录。
> 注意：§0-§5 的数字产生于动作条件因果对齐修复**之前**（cond[t] 旧为"进入帧"约定），与 §00 不可直接比较。

## 00. 第三轮：因果修复后的采样/损失变体对比（e8c8904 引入项的消融）

**设置**：统一 62M（token_dim 384 / dyn_layers 8 / enc_base 48 / shortcut_hidden 1024）、
batch 32 / seq 16 / bf16 / seed 42 / 5000 步 / camera_scale 32（全量 32 段 p95 重标定）、
holdout 3 段、n_eval_batches 8（已知单点噪声 ~±0.3dB，结论只取两个评估点一致的走势）。
每组 ~19 分钟，GPU 利用率 92-100%（吞吐 ~2270 帧/s，CPU 数据管线上限 ~4600 帧/s 未触及）。

**结果（holdout@5000，括号内为 @4000 佐证走势）**：

| 组 | gen−persist (dB) | EV(Δz) | IG | 开环@8 步优势 (dB) | val_flow |
|---|---|---|---|---|---|
| A 基线 | −0.17 (−0.65) | −0.092 (−0.166) | −0.417 | −0.89 (−1.32) | 0.140 |
| B motion_sample=4 | −0.12 (−0.04) | **+0.037 (−0.070)** | −0.308 | **−0.31 (−0.78)** | 0.158 |
| C delta_weight | **+0.01 (−0.29)** | −0.136 (−0.133) | −0.316 | −1.29 (−1.05) | 0.141 |
| D 组合 | −0.17 (−0.50) | −0.115 (−0.163) | **−0.181 (−0.178)** | −1.25 (−1.46) | 0.173 |

**判定**：
1. **B（运动量锦标赛采样）是动力学口径的胜者**：EV(Δz) 四组唯一转正（且两个评估点单调向好
   −0.166→−0.070→+0.037 的跨组最优走势），8 步开环差距缩到 −0.31dB（A 的 1/3）。
   采样层把梯度集中到高运动转移，改善的是"变化方向是否正确"，并随步数复利。
2. **C（|Δz|² 损失加权）赢在单步像素、输在动力学**：gen−persist 唯一非负（+0.01dB），
   genmean 四组最高（23.40dB），但 EV 恒负、开环最差——损失加权把单步预测push向大变化
   转移的像素均值，未改善变化的方向性；多步滚动下劣势放大。
3. **D（组合）无叠加收益**：采样已聚焦高运动后再做损失加权 = 对极端转移双重加权，
   val_flow 四组最高（0.173，欠拟合主体分布），除 IG 外各口径回落到基线水平。
4. **IG 四组恒负（−0.18 ~ −0.42）是横贯性异常**：难度头报告"带动作条件更难预测"，
   与动作语义被利用的预期相反。因走势在组间不服从其他口径的排序，怀疑是 IG 口径自身
   问题（零动作基线 out-of-distribution，或批内归一目标在小评估批下的符号不稳），
   列为下一步诊断项，暂不作为选型依据。
5. **对比修复前（§0，10k 步 gen−persist=−0.13±0.05）**：修复后基线 5k 步即达同量级（−0.17），
   预算减半；确认因果修复本身有效。

**选型**：后续离线预训练配方取 **B（--motion_sample 4）**，不叠加 delta_weight。
开环 rollout 与 EV 是与"取得 mine_stone 能力"最相关的口径（想象式规划依赖多步正确性），
优先级高于单步 PSNR。

**下一步**（服务主线 mine_stone，见 docs/activity_log.md 项目主线目标）：
① 用 B 配方 + 更长预算（≥15k 步，val_flow 未收敛）+ 更好动作覆盖的数据
（gaming-500-hours minecraft 子集转换，或 VPT all_6xx）重训；
② 诊断 IG 口径；③ 动态分辨率两处解耦（dynamics.pos_spatial 插值化、
tokenizer 解码器全卷积化）后在 128px 复验 B 的收益。

## 0. 第二轮：62M + bf16（`--amp bf16`，批量 32，GPU 利用率均值 85.5%）

模型 26.39M → **62.05M**（token_dim 384 / dyn_layers 8 / enc_base 48 / shortcut_hidden 1024），
bf16 autocast（三精度 60 步对照损失逐位一致，bf16 较 fp32 +26% 吞吐,见 dreamer.md §2.6），
batch 24→32 + workers 5 后稳态吞吐约 2100-2300 帧/s，GPU 利用率均值 85.5%（`scripts/sys_monitor.py` 实测,
显存峰值仅 4.7/23GB——还有继续加 batch/模型的空间）。

| 口径 | v1(26M,fp32) | **v2(62M,bf16)** |
|---|---|---|
| 离线 holdout psnr_gen @10k 步 | 22.39 dB | **22.84 dB** |
| 同批 psnr_persist | 23.03 dB | 22.70 dB |
| gen − persist（训练日志 8 batch 评估） | −0.64 dB | +0.14 dB |
| gen − persist（**32 batch/512 窗口配对复检**,不同窗口 seed） | — | **−0.134 ± 0.054 dB（未稳健超过）** |
| 离线重建上限 psnr_recon | 27.34 dB | 27.50 dB |
| 离线 10k 步耗时 | ~16 分钟 | 37.1 分钟（帧数 ×2:batch 32 vs 16/24） |
| 在线 held-out psnr_gen(best) | 16.67 dB | 16.70 dB（持平;瓶颈是在线数据量与随机策略,非容量） |
| 在线 24k env 步耗时 | 8.4 分钟(47 sps) | 5.8 分钟(~75 sps,JVM/区块已热) |

判定（以 32 batch 配对复检为准）：62M+bf16 把 gen−persist 差距从 −0.64dB 压到
**−0.13±0.05dB——追平但未稳健超过持续性平凡解**。训练日志里 @8.5k-10k 的 gen≥persist
是 8-batch 评估的抽样噪声（batch 间方差 ~0.3dB）,不作数;结论必须以大样本配对差为准。
20Hz、frame_skip=1 下相邻帧几乎不变,persistence 是极强基线;要真正越过,按预期收益:
① **frame_skip>1（Δt~U{1..4}）**——直接削弱 persistence、放大动作效应信噪比
（vpt_dataset 原生支持,jumpy prediction 正是其设计动机）;② 加数据与训练步数
（val_flow 未收敛）;③ 更大 dyn_layers。
在线侧持平——held-out 环境的生成质量受限于随机策略采集的数据多样性与
24k 步的数据量,继续扩容模型无收益,下一步应换有奖励覆盖的采集策略并加长在线采集。
在线阶段 GPU 利用率约 30%（环境步进主导）,抬升手段是异步采集（见 §5）。

按 Dreamer 4 路线在 Minecraft 上完成世界模型的**离线（VPT 真数据）与在线（CraftGround 交互流）**
两种方式训练与评估。模型 26.39M 参数（token 网格 4×4、token_dim 256、dyn_layers 4），
64px 观测，评估口径见 `knowledge/dreamer.md` §2。

## 1. 离线：VPT find-cave 真数据（`train/minecraft/train_dreamer4`）

- **数据**：32 段 BASALT find-cave 承包商数据（mp4+jsonl → 契约格式，
  `tests/download_vpt_data.py`；camera_scale=29 按 |dx| p95 标定），末 3 段为 holdout。
- **训练**：10000 步（batch 16 × seq 16），共约 16 分钟（4000 步首轮 5.7min + 续训 10.1min），
  吞吐约 2500-2900 帧/s。
- **结果（holdout@10000）**：

| 指标 | 数值 | 含义 |
|---|---|---|
| psnr_gen（4 步流生成下一帧） | **22.39 dB** | 从训练初 ~11dB 一路上升 |
| psnr_persist（复读上一帧基线） | 23.03 dB | gen 尚差 0.64dB |
| psnr_recon（tokenizer 重建上限） | 27.34 dB | 仍在上升 |
| val_flow | 0.107 | 全程单调下降，未收敛 |

- **判定**：动力学显著在学（gen 相对冷启动 +11dB、val_flow 未平台化），但 10k 步内
  少步生成还没有超过持续性基线——20Hz 相邻帧几乎相同，persistence 在 PSNR 口径下是强基线。
  val_flow 未收敛说明瓶颈是训练预算而非容量；下一轮优先加步数/数据，其次升 dyn_layers。

## 2. 在线：CraftGround Minecraft 1.21（`train/craftground/train_dreamer4`）

- **采集**：3 env（env 2 held-out 只评估不训练），随机均匀策略（27 离散动作），
  GPU 无头渲染（Xorg :1，NVIDIA L4），**47 sps** 聚合吞吐；24000 env 步共 8.4 分钟。
- **热启动**：`--init` 离线 best.pt（动作接口 22 维连续 → 27 维 one-hot，
  action_proj/reward/cont 重新初始化，其余权重迁移）。
- **结果（held-out env）**：

| 指标 | 数值 | 含义 |
|---|---|---|
| psnr_gen | **16.67 dB**（best） | 末段稳定在 ~15dB |
| psnr_persist | ~17.1-17.6 dB | gen 差约 1-2dB |
| psnr_recon | ~19-20 dB | 在线画面（随机转头）重建上限低于离线 |
| reward NLL / MAE | ≈ 0 | ⚠ 平凡解:随机策略下奖励几乎全 0 |
| cont_acc | 1.000 | ⚠ 平凡解:窗口内 done 极稀 |

- **判定**：在线世界模型管线（采集→回放→更新→held-out 评估）全链路工作；
  生成质量趋势同离线（逼近但未超 persistence）。reward/cont 头的完美数字**没有信息量**
  ——随机策略几乎采不到非零奖励，头学到的是"恒 0/恒 1"。要让这两个头有意义，
  需要有奖励覆盖的采集策略（接 PPO+AD 策略采集，或成就课程）。

## 3. 辅助结果（同轮完成）

- **离线 BC 线**（`net/bc` + `train/minecraft/train_bc`，冻结 DINOv3 + 3.77M 可训练）：
  32 clips、6000 步。holdout 相机 bin top-1 = 0.813（多数 bin 基线 0.803、持续性 0.828）、
  按键 micro-F1 = 0.954（持续性 0.949）——键位略超持续性基线，相机未超；
  6 clips 时明显过拟合（500 步后 holdout 变差），32 clips 后缓解。
  行为克隆要在这两个基线上拉开差距，数据量需要再上一个量级。
- **在线随机策略 baseline**（PPO+AD 管线，16k 步）：曾解锁 3/36 成就；
  近 8 局 per-episode 成功率 flower=12%、seeds=50%。这是后续任何在线策略学习的对照下限。

## 4. 环境与工程要点（可复用）

- **Colab L4 无头 GPU 渲染**：镜像自带匹配版本的 NVIDIA Xorg 驱动
  （`/usr/lib64-nvidia/xorg`，与内核驱动同 580.82.07），标准仓库的
  xserver-xorg-video-nvidia 版本不匹配、**不要装**。要点：xorg.conf 里
  `ModulePath "/usr/lib64-nvidia/xorg/modules"` + 实际 BusID（`lspci` 换算），
  L4 **不支持** `Option "UseDisplayDevice" "None"`（删掉即可）；
  `/usr/lib64-nvidia` 需加入 ld.so.conf 使 libGLX_nvidia 可被 glvnd 发现。
  实测 glxinfo renderer = "NVIDIA L4/PCIe/SSE2"，craftground 3 env 聚合 47 sps。
- **CraftGround 依赖**：Java 21 必须装**完整版** `openjdk-21-jdk`
  （headless 版缺 AWT，gradle 的 CMake FindJNI 报 "missing: AWT"）。
  首次启动 gradle 构建约 4-8 分钟，且从 resources.download.minecraft.net
  下载资源可能瞬时失败——直接重试即可（缓存续传）。
- **VPT 真数据**：`openaipublic.blob.core.windows.net/minecraft-rl` 公开可达，
  单段 100-300MB;jsonl 与帧 1:1（20Hz），`isGuiOpen`→gui 标记，
  鼠标 `buttons` 0/1 → attack/use。find-cave 子集 |dx| p95≈29（据此定 camera_scale）。

## 5. 下一步（按收益排序）

1. **离线加预算**：val_flow 未收敛,gen 与 persist 差 0.64dB——首选加训练步数与 clip 数
   （当前 32 段仅约 2.7 小时游戏时长）,预期 gen 过基线。
2. **在线换采集策略**：随机策略喂不出 reward/cont 信号;接 PPO+AD 策略或
   ε-贪心成就课程后,reward 头评估才有意义。
3. **策略阶段**：Dreamer4 的想象 actor-critic / 策略蒸馏（`net/dreamer4/agent.py`
   已有 actor/critic 头结构,训练待补）。
