# Minecraft Dreamer4 离线 + 在线首轮训练结论（2026-07-02，Colab L4）

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
