# AI 助手活动日志

> 按用户要求记录"助手做了什么"的操作流水（含环境配置与实验过程），最新条目在最下方。
> 分析结论仍按规范沉到 `knowledge/`，此处只记过程与事实。

## 2026-07-02（Colab L4，会话 1）

### 环境配置
- `python install_env.py --dreamer --ppo-ad --minecraft --dev`（Colab demo 标准组合 + minecraft）。
  - **故障 1**：headless 系统依赖的 `apt-get install` 在 `keyboard-configuration` 包弹出交互式
    debconf 键盘布局询问，挂起约 20 分钟。处置：杀掉进程树，`DEBIAN_FRONTEND=noninteractive
    dpkg --configure -a` 后重装系统依赖成功。**教训：install_env.py 的 apt 调用应设
    `DEBIAN_FRONTEND=noninteractive`（待修）。**
  - **故障 2**：`minecraft` extra 的 `minerl==0.4.4` 依赖 gym<0.20，gym 0.19 与新 setuptools
    不兼容（`extras_require` schema 错误），uv 构建失败。处置：离线 Dreamer4 管线不 import
    minerl，改装 `.[crafter,dev,dreamer,headless,ppo-ad]` + pillow，跳过 minerl。
  - 冒烟：`from net.dreamer4 import WorldModel` 与 `import crafter` 通过。
- 站点包里存在第三方 `tests` 包，遮蔽仓库 `tests/` 命名空间包 ⇒ `python -m tests.download_vpt_data`
  失效；改用 `PYTHONPATH=/content/repo python tests/download_vpt_data.py` 直跑。
- 设备监控：`scripts/sys_monitor.py --interval 5 --csv runs/logs/sys_monitor.csv` 已后台常驻。
- 定期保全：会话内每 10 分钟自动 commit+push 一次（防 Colab 中断丢工作）。

### 数据
- `tests/download_vpt_data.py --index find-cave-Jul-28 --n 32 --out runs/data/vpt_findcave`
  后台下载中（BASALT find-cave 承包商数据，与上一轮结论文档同源同规模）。
  首 2 段标定：|dx| p95=58 ⇒ 建议 camera_scale≈33（全量下完后按汇总分位数定，上轮为 29）。

### 利用率标定（scripts/sys_monitor.py + 7 配置 240 步探测）
- 吞吐被 CPU 数据管线封顶 ~4600 帧/s：batch 32/64/128 稳态帧率同为 4.5-4.7k，
  workers 9→11、clip_cache 4→8 均无增益。
- GPU 利用率随每步计算量走：26M 模型 batch 32→22%、64→71%、128→85%。
- `motion_sample 4` 的 4 倍候选解码开销被闲置 CPU 完全吸收（吞吐 4617 vs 4686 帧/s）。
- **结论：占满 GPU 用"加大模型"而非"加大 batch"**——62M 结构（token_dim 384/dyn_layers 8/
  enc_base 48）+ batch 32 + bf16 即为上轮实测 85% 利用率组合，四组实验采用之。

### 实验计划（进行中）
- 目标：对比 e8c8904 引入的修改方式的效果，四口径评估（psnr_gen−persist、EV(Δz)、IG、
  8 步开环 rollout 优势）。
- 矩阵（统一预算 5000 步 / batch 32 / seq 16 / bf16 / seed 42 / camera_scale 32（全量 p95 标定）/
  62M 结构；holdout_n=3，eval_interval=500）：
  - A 基线（对齐修复后，motion_sample=1，无 delta_weight）
  - B `--motion_sample 4`
  - C `--delta_weight`
  - D `--motion_sample 4 --delta_weight`
- 结果与判定将写入 `knowledge/conclusion_minecraft_dreamer4_run.md`。

### 实验结果流水（holdout@5000，n_eval_batches=8）
- **A 基线**（07:10-07:28，GPU 92-100%，~2300 帧/s）：gen=22.59 persist=22.76（差 −0.17dB）
  genmean=23.13 recon=26.95 val_flow=0.1395（未收敛）EV(Δz)=−0.092 IG=−0.417
  开环@8 步优势 −0.89dB。初步观察：因果修复后 5k 步即达上轮 10k 步水平；IG 为负值得警惕
  （无动作反而更易预测？待四组齐后统一分析）。
- **B motion_sample=4**（07:28-07:47）：gen=22.64 persist=22.76（差 **−0.12dB**）genmean=23.10
  recon=26.45 val_flow=0.1578 **EV(Δz)=+0.037（转正）** IG=−0.308 开环优势 **−0.31dB**。
  相对 A 全面向好：EV 首次为正、开环差距缩小 2/3。
- **C delta_weight**（07:47-08:06）：gen=22.77 persist=22.76（差 **+0.01dB，四组首个非负**）
  genmean=23.40 recon=26.81 val_flow=0.1414 EV(Δz)=−0.136 IG=−0.316 开环优势 −1.29dB。
  口径分化：单步 PSNR 上首次追平/略超 persistence，但 EV 为负、开环劣于 A——损失加权
  赢在"下一帧像素"，没赢在"变化方向正确"，与 B 恰好互补，D（组合）是关键判据。
- **D 组合**（08:06-08:25）：gen=22.59 persist=22.76（差 −0.17）EV=−0.115 IG=−0.181
  开环优势 −1.25dB val_flow=0.173（四组最高）。无叠加收益。
- **四组齐，判定与选型写入 knowledge/conclusion_minecraft_dreamer4_run.md §00**：
  胜者 B（motion_sample=4），EV 唯一转正 + 开环差距缩至 1/3；IG 恒负列为诊断项。
- 新数据源调查：HF `markov-ai/gaming-500-hours` minecraft 子集 75 段（~30h，1080p/30fps，
  逐帧 OS 输入事件带 appName 过滤）。转换风险：FPS 鼠标捕获模式绝对坐标差分待实测。

### 高清语料吃法设计（用户指示：先思考再转换）
- 实测 L4 视频引擎：NVDEC 解码 1080p 25.9×实时（~780 帧/s、零 CPU）；
  NVDEC+scale_cuda+NVENC 全 GPU 转码 25.6×实时 vs libx264 CPU 路径 10.1×（且吃满 12 核）。
  NVENC 同段体积 10.3MB vs x264 6.7MB，cq 待标定。
- 发现：训练 `--img_size` 默认 64——四组实验实际输入 64²，仅占 1080p 像素 0.2%。
- 设计结论写入 `knowledge/design_gaming500_hd_pretrain.md`：分层吃法（tokenizer 吃原生
  密度裁剪、dynamics 吃全时长 token cache）、一次解码两路落盘、img_size 提 128、
  磁盘预算与四阶段执行时序。转换暂停，待用户确认方案。

### 学习式注意 v0 落码 + 128² 基线启动
- `WorldModel.loss` 新增 `hard_weight`(OHEM 硬样本重加权,模型自身流误差 detach 温度化,
  [0.25,4] 有界,flow/sc 同权);train_dreamer4 增 `--hard_weight` 透传。冒烟通过
  (含与 delta_weight 组合)。本轮 128² 长跑未开启(保持 B 配方纯净口径,留给下轮消融)。
- **修 bug**:ConvDecoder 输出分辨率 = dec_min_res×16 与 obs_shape 脱钩,img_size=128 时
  重建目标 64² vs 输入 128² 形状错——train_dreamer4 现按 img_size//16 反推 dec_min_res,
  并断言 img_size 为 16 倍数。
- 转换器升级:probe_nvenc 自检 + cut_video 走 NVDEC 解码/NVENC 编码(不占 CPU/SM);
  `--crop-stream` 每段附加输出 1080p 原生像素运动能量最大 256² 窗口(积分图搜索,
  排除任务栏区),无 jsonl 配对不进动力学训练,专喂 tokenizer;会话级断点续转。
  旧命名(无游戏前缀)的 17 段已删避免重复偏置,新版转换器后台重转中(--purge-raw)。
- **128² 基线长跑启动**(runs/mc_d4_b128,PID 112078):B 配方 motion_sample=4,
  token 网格 8×8=64(隐码容量×4,解闸 1+2),646M 参数(~600M 是解码器平铺 Linear,
  分辨率耦合点#2 的已知代价),batch 16 bf16,15k 步,~262 帧/s 预计 4-5h。
- 冒烟目录 runs/mc_d4_smoke128 已按用户要求清除。
- 用户拍板 W/C/A 三模型架构,设计入库 knowledge/design_wca_agent.md
  (A 硬约束:绝对禁止直接吃原始内容输入,只吃 W 隐状态)。

### 项目主线目标（用户 2026-07-02 明确）
- **最终目的：快速学会 Minecraft 动作，达成 mine_stone（Stone Age）及以上成就。**
  世界模型与变体对比是工具不是目的；路线为 离线世界模型（选出最优配方）→ VPT 动作先验（BC）
  → CraftGround 在线 achievement_rewards（mine_wood→mine_stone 课程）+ PPO。已确认
  `train/craftground/achievements.py` 含 mine_stone 依赖链、reward.py 有成就奖励与稠密内在奖励通道。

### 脑内推演可视化(用户要求"看看预测长啥样")
- 新增 tests/viz_rollout.py:holdout 上 K 步开环 rollout → [GT|DREAM|PERSIST] 三联
  对比视频(libx264,浏览器可播)+ 每样本胶片图(3 行×horizon 列)。
- 已渲染两个 ckpt:runs/viz/b128.mp4(128²,step 1500 早期)与 runs/viz/b64_ms4.mp4
  (64² B 配方,step 4000)。目视结论:梦境保留场景构图(草地/天空/HUD 位置正确)
  但纹理糊(L2 频谱偏置+隐码容量+训练早期三因素叠加),随步数漂移加剧
  (即 roll_adv 测的误差复利);64²/5k 步比 128²/1.5k 步稳定,分辨率换训练量。

### 128² 基线中途流水(runs/mc_d4_b128)
- holdout@3500:**gen=22.26 > persist=21.64(+0.62dB,单样本单步口径首次转正)**
  genmean=22.84 recon=26.28 val_flow=0.2354 **EV(Δz)=+0.087**(64² B@5k 为 +0.037)
  IG=-0.225 开环@8 步优势 -1.15dB(仍负,但预算内:recon-persist≈4.6dB 空间)。
  初步支持"隐码容量×4(S=64)解闸 2"的设计判断;开环与 IG 待 15k 步终评再判。
- **收割点改 10k 步**(用户问"两小时会不会太长"):LR 仅 warmup 无退火,中途收割零损失;
  10k×batch16×seq16=256 万帧,恰与 64² B@5k(batch32)同帧预算,是"容量×4 值不值"的
  最干净对比切点;单步转正与 EV 翻倍已落袋,10k→15k 只打磨 checkpoint(Colab 死即失,
  存活价值低)。看门狗后台监听日志,step 10000 评估存档后自动杀训练进程。
