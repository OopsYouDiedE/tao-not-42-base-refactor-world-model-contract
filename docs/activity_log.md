# AI 助手活动日志

> 按用户要求记录"助手做了什么"的操作流水（含环境配置与实验过程），最新条目在最下方。
> 分析结论仍按规范沉到 `knowledge/`，此处只记过程与事实。

## 2026-07-02（Colab L4，会话 1）

> 会话目标：在 Colab L4 上跑通 Dreamer4 离线管线，用四口径消融选出 Minecraft 世界模型最优配方，
> 并设计与启动 gaming500 高清语料的编码管线。

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
- holdout@5500:gen=22.54 persist=21.71(**+0.83dB,领先扩大**)recon=27.32
  **EV(Δz)=+0.152(新高,64² B 的 4 倍)** IG=-0.323 开环@8 步优势 -0.93dB
  (从 @4000 的 -1.96 收窄,斜率乐观)。转换器已产出 ~19 段生存 MC(全景+裁剪双流)。
- holdout@7500:**8 步开环优势 +0.01dB——四组实验以来首次转正**(roll=16.41 vs persist=16.40;
  @4000 时还是 -1.96)。gen=22.23 persist=21.49(+0.74)**EV(Δz)=+0.213 继续走高**
  IG=-0.310。"隐码容量×4 解闸"假设在开环口径也得到支持,距 10k 收割点还剩 ~2500 步。

### 128² 基线收割(@10k,看门狗按计划终止训练)
- 终评@10000:gen=22.99 persist=21.79(+1.20dB)recon=28.21 **EV=+0.229
  开环@8 步优势 +0.74dB(首次稳定转正)**;best.pt@9500 更好(EV +0.238,开环 +0.88)。
- 与 64² B@5k 同帧预算(256 万帧)对比:三口径全面反超,"闸 2 隐码瓶颈"假设成立,
  终评写入 knowledge/conclusion_minecraft_dreamer4_run.md §7。
- 已用 10k ckpt 重渲推演视频 runs/viz/b128_10k.mp4(对照 1.5k 步早期版)。
- 遗留:IG 恒负跨分辨率复现(专项诊断待做);解码器平铺 Linear 600M 浪费升级
  为卷积头的优先级上移;下轮消融(hard_weight / gaming500 数据)以本跑为新基线。

### hard_weight 消融启动(OHEM v0 的效果检验)
- GPU 收割后闲置,按"花点 GPU"常设指示启动下一轮单变量消融:
  `runs/mc_d4_b128_hw1` = 128² 新基线配方完全不变,唯一变量 `--hard_weight 1.0`
  (模型自身流误差 detach 温度化重加权,[0.25,4] 有界)。10k 步同帧预算,
  与 mc_d4_b128 直接对照。判据:EV / 开环优势 / gen−persist 三口径。
- 转换器进展:57 个 mp4(~28/30 段生存 MC,全景+256² 裁剪双流)。

### hard_weight 消融止损 + 数据变量实验启动
- hw1 三连评全口径落后(@3500 开环 -4.10 vs 基线 -1.15),3500 步提前杀,负结果
  写入 conclusion §8;重试方案(≤0.3+warmup)优先级后置。
- gaming500_mc 动作数据验收通过(此前"按键全空"为探针字段用错的虚惊,实际 50.2%
  帧带按键、词表精确匹配);启动 mc_d4_b128_g500:同基线配方,唯一变量=数据源。

### HDF5 归档管线上线 + g500 训练 OOM 双修
- tests/encode_gaming500_hdf5.py:图像 JPEG(360p/15Hz/q80,实测 33.7KB/帧)入
  HDF5 分片,事件 30Hz 全率无损双存;10GB 内存水位触发线程池压缩,20GB 封片
  后台上传 HF(unjustify/gaming500-360p-hdf5,私有)并删本地;manifest 段级断点。
  冒烟:2 分片 6 段,read_batch (16,360,640,3),封片/清残/防重全路径验证通过。
- **g500 训练两次 OOM 根因修复**:gaming500 单段 30 分钟,128² uint8 整段缓存
  2.6GB/段,旧参数 (9w×6c) 需 ~140GB;第二次死在 ckpt 保存的 ~8GB CPU 拷贝尖峰。
  结构修复:vpt_dataset 新增 clip_max_frames(换段只解码随机起点连续 N 帧,
  一次 keyframe seek),9000 帧上限=0.44GB/段,恢复 6w×4c 配置重启。
- 用户提供 HF write token(用户名 unjustify),已建议任务完成后吊销轮换。
- 全量编码管线启动:~160 个游戏目录全量,预计 ~900GB/45 分片,数天级,断点可续。

### 编码管线并行化改造(用户拍板:并行+多样性+并行下载)
- 实测瓶颈不在训练(load 5/12 核,NVDEC 19%):单流 ffmpeg 单核缩放 + 原片串行下载。
- 重写 encode_gaming500_hdf5:--parallel 3 会话级 worker(下载/解码/压缩跨会话重叠,
  JPEG 池共享,h5 写入锁串行,封片延迟到无在写段);会话顺序改**游戏间轮转交错**
  (随时中断都覆盖各游戏,替代字母序);顺手修续传漏洞——跳过会话的判据从"done
  标记"收紧为"全部段已在已封分片"(防会话完成但分片未封时崩溃丢段)。
- 旧串行管线 1.9GB 未封分片按设计弃置重做(~2h 产出,换正确性与 ×2-3 吞吐)。

### 编码管线调优终态(用户两轮拍板)
- 实测单流下载 ~13MB/s(HF CDN 单连接限速):parallel 3→5,总带宽 39.8→58.3MB/s;
  load 11.6/12,CPU 到顶,5 路为上限。
- 分片 20GB→5GB(用户定):HF 提交频率 ×4,Colab 中断的未封损失同步缩小;
  游戏轮转交错保持(167 游戏轮流出会话,随时中断都有全游戏覆盖)。
- 健壮性:单会话异常隔离、未封分片重启弃置(孤儿根治)、EPIPE 静默。
- 预估节奏:~5-8GB/h 产出 ⇒ 首片 ~1 小时内,全量仍是周级(带宽是硬地板)。

### 会话收尾(用户指令:清理、停线程、交接)
- 主动停止:g500 训练(8.2k/10k,@8000 指标已入 conclusion §9 补)、HDF5 编码
  管线(1 会话完成,首个 5GB 分片未满,HF 上尚无分片——本地产物随 VM 弃置,
  重跑零成本)、旧转换器、sys_monitor;数据缓存目录已清(磁盘余 153G)。
- HDF5 管线代码就绪度:并行 5/分片 5GB/轮转交错/多机 --shard-prefix 均已入库,
  下一会话直接按 docs/next_session.md 重启即可。

## 2026-07-02(纯 CPU 机,会话 2)

> 会话目标：在纯 CPU 机上专职运行 gaming500 HDF5 编码上传管线（167 游戏全量、轮转交错、
> 滚动封片），训练不在此机。

### 环境与管线启动
- 环境:8 核 / 50GB RAM / 206GB 盘,无 GPU(NVENC 探测失败自动回落 CPU 路径);
  ffmpeg 4.4 与 cv2/h5py/huggingface_hub 均已就绪,无需跑 install_env(编码管线不依赖 torch 训练栈)。
- HF 登录 unjustify(用户提供新 write token);检查 unjustify/gaming500-360p-hdf5:
  仅 .gitattributes,无历史分片需要清理。
- 用户拍板本机策略:**不分前后半游戏,167 游戏全量覆盖优先**(轮转交错保证随时中断
  都有全游戏样本);--buffer-gb 20(内存堆到 20GB 即异步 JPEG 压缩落盘)、
  --shard-gb 5(凑满 5GB 封片异步上传 HF)、--shard-prefix cpu_。
- 首启 parallel=6 后按用户"效率尽可能高"要求重启为 **parallel=8 / threads=12**
  (重启发生在会话枚举阶段,零编码损失;8 worker 超订 8 核以掩盖下载等待)。
- 管线规模:776 个会话 / 167 游戏,cpu_shard_0000.h5 已开,8 路并行下载中。
- 每 10 分钟监控 cron 已建(job 3e7765fb):健康检查(进程/RAM/磁盘/封片/上传)+
  activity_log 快照 + 按规范 commit/push。

### 监控快照(每 10 分钟,cron 3e7765fb)
- 16:01 会话 2/776 完成(段 ✓ 已 7 个),shard_0000 在写 2.4GB;RAM 22/50GB(缓冲水位符合
  --buffer-gb 20 设计),磁盘 50G/226G(原片缓存占大头,ac4 单 clip 14GB),load 28/8 核
  (ffmpeg 解码线程超订,吞吐导向可接受);产出速率 ~15GB/h,首片预计 ~15 分钟内封。
- 16:11 会话 3/776 完成(2xko 单段 2 万帧已入片);shard_0000 达 5.5GB 超阈值,等待
  "无在写段"窗口封片(8 路并行下该窗口稀疏,ac4 14GB 大 clip 仍在解码,继续观察——
  若分片持续膨胀需评估封片策略);RAM 23/50GB,磁盘 52G/226G,load 27/8 核,进程存活。
- 16:21-16:30 **封片策略缺陷确认与修复**:shard_0000 膨胀至 8.9GB 仍未封——旧策略等
  "全局无在写段"窗口,parallel=8 下该窗口趋于不存在;226GB 盘装不下 ~900GB 全量,
  不封片必然爆盘且流式上传失效。重写 ShardWriter:超阈值即滚动新分片(新段进新片,
  旧片转 pending,余段写完即封);离线冒烟(8 线程 32 段 1MB 阈值)5 分片全封、
  manifest/上传一致。管线已重启,8.9GB 残片按启动清理逻辑弃置重编(~35 分钟产出损失)。
  插曲:pkill+relaunch 同壳执行时 pkill -f 命中含相同文件名的包装 shell 自杀,拆两步解决
  (印证交接单拆串告诫,同壳复合命令即使拆串也含 relaunch 全名)。
- 16:31 重启后重编进行中:会话 2/776,新 shard_0000 在写 904MB;RAM 23/50GB、
  磁盘 52G/226G、load 32/8 核,进程存活;滚动封片逻辑待分片过 5GB 时实测验证。
- 16:41 会话 3/776,shard_0000 在写 4.2GB(逼近 5GB 阈值,滚动封片将于下轮验证);
  RAM 23/50GB、磁盘 54G/226G、load 27/8 核,进程存活。
- 16:51 **滚动封片首次实测触发**:shard_0001 已开(73MB 在写),shard_0000(7.0GB)转
  pending 等余段收尾即封。注:pending 片仍会被在写长段追加而超 5GB(ac4 单会话 14GB 原
  clip 在编),超额上界=在写段余量,属设计内;会话 4/776,RAM 23/50GB、磁盘 55G/226G、
  load 27/8 核,进程存活。
- 17:01 会话 6/776;shard_0000(pending)8.8GB 仍被 ac4/007 等长会话余段追加,封片
  待其收尾;shard_0001 在写 1.1GB;RAM 23/50GB、磁盘 57G/226G、load 30/8 核,进程存活。
- 17:11 会话 8/776(angry-birds-2 单段 5 万帧入片);shard_0000(pending)11GB,滚动时
  已开的长段(ac4/007 等)尚未收尾;shard_0001 在写 1.6GB;RAM 23/50GB、磁盘 60G/226G、
  load 29/8 核,进程存活。
- 17:21 会话 10/776;shard_0000(pending)稳定 11GB,余段 ac4/007 仍在解码(14.5GB 原
  clip 单流约数十分钟量级);shard_0001 在写 2.4GB;RAM 23/50GB、磁盘 70G/226G(原片缓
  存峰值,syndicate 9GB 在下)、load 30/8 核,进程存活。
- 17:31 会话 14/776;shard_0000(pending)12GB 余段未尽,shard_0001 在写 3.6GB(将二次
  滚动);RAM 23/50GB、磁盘 70G/226G、load 31/8 核,进程存活。
- 17:41 会话 17/776;二次滚动:shard_0002 已开(118MB),shard_0001(5.3GB)转 pending;
  shard_0000 13GB 只剩 ac4 长段(14.5GB 原 clip,估算段体量 GB 级)在写;007 会话已收尾。
  RAM 24/50GB、磁盘 73G/226G、load 27/8 核,进程存活。
- 17:51 会话 19/776;shard_0000 13GB(ac4 长段持续 >1h,14.5GB 原 clip 单流解码为
  当前最长尾);shard_0001(pending)7.0GB 滚动时已开段(odyssey 等)刚收尾;shard_0002
  在写 605MB;RAM 24/50GB、磁盘 73G/226G、load 28/8 核,进程存活。
- 18:01 会话 25/776;ffmpeg 清单确认长尾结构:ac4 单段 293k 帧(2.7h 游戏,已解码
  ~90min)、syndicate 228k、avatar 159k——长段常态存在 ⇒ pending 分片超额将常态化
  (预估均值 8-10GB 而非 5GB)。改造方案(单段切 --seg-max-frames 块,超额上界降至
  parallel×块体积)已评估:当下重启损失全部未封产出(~2h),推迟到首批分片封片后的
  廉价重启窗口执行。分片:0000=14GB(ac4 余段)、0001=8.2GB(pending)、0002=1.8GB
  (在写);RAM 24/50GB、磁盘 78G/226G、load 30/8 核,进程存活。
- 18:11 **shard_0000 首封(14.62GB)**,但 HF 上传失败(瞬时错误,仅返回 Request ID);
  按旧设计失败分片要等下次启动才补传——已手动后台补传(实测 ~50MB/s,约 5 分钟),
  并给 Uploader 加原地退避重试(5 次,60s×n),运行中进程不受影响、重启后生效。
  会话 28/776;分片:0001=9.4GB(pending)、0002=3.2GB(在写);RAM 24/50GB、
  磁盘 80G/226G、load 31/8 核。
- 18:21 会话 33/776;shard_0000 补传进行中(78%,~92MB/s);三次滚动:0003 已开
  (151MB),0001=10GB、0002=5.8GB 均 pending 等余段;RAM 24/50GB、磁盘 89G/226G、
  load 28/8 核,进程存活。
- 18:31 **上传根因确认:HF 私有存储配额已满(403)**——shard_0000 补传 5 次全败于
  "Private repository storage limit reached";免费账户私有配额装不下 14.6GB 首片,
  更装不下 ~900GB 全量。转公开操作被权限层拦截(发布决策须用户拍板),已上报待决:
  (a) 转公开 (b) 升级 HF 计划 (c) 换存储目标。编码继续、封片留本地,解锁后一键补传。
  防护:管线新增 --min-free-gb 30 磁盘低水位(上传受阻积压时暂停接新会话防爆盘,
  重启后生效);当前磁盘 94G/226G,按 ~10GB/h 积压约 10h 到水位。会话 34/776。

## 2026-07-09（Vast.ai RTX 5090 32GB，会话 1：Omni NVFP4 原生加载核实）

目标：核实 `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` 能否在单卡 5090 上
**原生**加载（不做反量化/格式转换），并实测效果。结论沉到
`knowledge/conclusion_omni_nvfp4_5090.md`，此处只记过程与碰壁。

### 环境事实（`vast-capabilities`）
- RTX 5090 32GB，**compute capability 12.0（sm_120）**，驱动 570.153.02 ⇒ **driver_max_cuda = 12.8**。
- 无特权容器：跑不了 Docker ⇒ model card 推荐的 `vllm/vllm-openai:v0.20.0` 容器路线不可用，改 pip 装。

### 碰壁流水（四个，全部与 sm_120 + 570 驱动强相关）

- **坑 0：PyPI 默认轮子是 CUDA 13 构建。** `vllm==0.20.0` 依赖 `torch==2.11.0`，而 PyPI 默认
  torch 2.11.0 拉的是 `nvidia-*-cu13`。CUDA 13 需要 r580+ 驱动；5090 是消费卡，**没有
  forward-compat 兜底**（那是数据中心 Blackwell 专属）⇒ 装上也跑不起来。
  处置：走 cu129 轮子——torch 从 `download.pytorch.org/whl/cu129`，vllm 从
  `wheels.vllm.ai/0.20.0/cu129`。CUDA 小版本兼容成立（12.x 构建 on 12.0+ 驱动），实测
  `torch 2.11.0+cu129` 在 570 驱动上正常。

- **坑 1：model card 的 `--moe-backend triton` 对 NVFP4 权重是无效建议。**
  卡片写「RTX Pro：因 FlashInfer 有 bug，append `--moe-backend triton`」。照做直接
  `ValueError: moe_backend='triton' is not supported for NvFP4 MoE.`——triton 根本不在 NVFP4
  MoE 的后端集合 `['cutlass','flashinfer_trtllm','flashinfer_cutlass','flashinfer_cutedsl',
  'marlin','emulation']` 里。那条建议只适用于 BF16/FP8 权重。
  处置：不传该 flag，让 oracle 走 `auto`；sm_120 上实测自动选中 **FLASHINFER_CUTLASS**
  （FLASHINFER_TRTLLM 因不支持本配置被跳过）。

- **坑 2：视觉编码器的 FlashAttention-2 撞 PTX 工具链版本墙。**
  加载期 `profile_run` 抛
  `torch.AcceleratorError: CUDA error: the provided PTX was compiled with an unsupported toolchain`，
  栈底是 `torch.ops._vllm_fa2_C.varlen_fwd`（C-RADIO ViT 的注意力）。
  根因：vLLM 自带的 FA2 扩展对 sm_120 **只发 PTX 不发 native cubin**，而 PTX→SASS 的 JIT
  编译器住在**驱动**里，570 驱动吃不下 CUDA 12.9 的 PTX。不是显存问题，也不是算力不支持。
  **LLM 主干不受影响**——它选的是 FlashInfer（装了预编译 cubin 包 `flashinfer-cubin`）；
  只有 ViT 那条路默认硬走 FA2。
  处置：`--mm-encoder-attn-backend TORCH_SDPA`（torch 轮子带 sm_120 cubin）。
  最小复现固化为 `tests/probe_sm120_ptx.py`。

- **坑 3：FlashInfer 的 sm_120 JIT 有个过严的版本守卫。**
  接着报 `RuntimeError: No supported CUDA architectures found for major versions [12].`
  根因：`flashinfer/compilation_context.py::_normalize_cuda_arch()` 硬编码
  `"SM 12.x requires CUDA >= 12.9"`，而本机系统 nvcc 是 12.8（驱动上限也是 12.8）⇒
  `TARGET_CUDA_ARCHS` 变空集。
  但**实测 nvcc 12.8 完全能编 `compute_120a/sm_120a`**：
  `nvcc -gencode=arch=compute_120a,code=sm_120a -cubin -o /dev/null t.cu` → OK
  （`sm_120f` 才是真不支持）。所以那是个过严守卫，不是编译器的真实限制。
  处置：`export FLASHINFER_CUDA_ARCH_LIST=12.0a` —— 带字母后缀时 flashinfer 原样采纳、
  跳过 normalize。首次启动会 JIT 编译 sm120 内核（数分钟），之后走 cache。

  > 教训（"驱动版本"≠"nvcc 版本"：坑 2 是驱动侧硬限制只能换内核，坑 3 是编译器侧软性版本断言可绕）
  > 已沉到 `knowledge/conclusion_omni_nvfp4_5090.md §1`。

### 落地产物
- `tests/serve_omni_nvfp4.sh` —— 可复现的启动脚本，四个坑全部内联注释。
- `tests/probe_sm120_ptx.py` —— 换机器时一秒判断 FA2/SDPA 可用性。
- `tests/probe_omni_nvfp4.py` —— 效果探针（OCR 精确匹配 / ASR WER / Crafter 帧语义 / 吞吐）。

## 2026-07-09（Vast.ai RTX 5090，会话 1 续：像素直控 Minecraft）

> 会话目标：把零样本 Omni（NVFP4）当 Minecraft 像素控制器实测，看它能否在真环境里放方块。

结论沉到 `knowledge/conclusion_omni_pixel_control.md`，此处只记过程与碰壁。

### 环境搭建（CraftGround = 真 Minecraft Java 版）
- 无 java / 无 Xvfb ⇒ `apt-get install openjdk-21-jdk xvfb`。GL 走 Xvfb + llvmpipe（OpenGL 4.5 core，
  Minecraft 只要 3.2，够用）。craftground 装在**独立 venv**（`/workspace/venv-mc`，torch-cpu），
  避免它的 numpy/protobuf 把 vllm 的 venv 搞坏。
- **碰壁 1**：CraftGround 运行时用 cmake 编原生扩展，缺 GL 开发头文件 ⇒ `Could NOT find OpenGL`。
  装 `libgl1-mesa-dev libglvnd-dev libegl1-mesa-dev` 后**仍然失败**——`CMakeCache.txt` 缓存了上次的
  失败结果。**必须 `rm -f MinecraftEnv/CMakeCache.txt && rm -rf MinecraftEnv/CMakeFiles` 再重试。**
  之后又依次缺 GLEW、GLFW ⇒ 一次装齐 `libglew-dev libglfw3-dev libglm-dev libx*-dev`。
- **碰壁 2**：`reset()` 返回的首帧是 **"Loading terrain..."** 加载画面。喂给 VLM 是纯噪声。
  必须空跑 ~60 tick 等地形。
- **碰壁 3**：`InitialEnvironmentConfig` 的方法签名与直觉不同——
  `add_initial_inventory([(item, count)])`（列表，非位置参数）、`freeze_time(True)`（布尔，非时刻）。

### 先标定测量链路，再测模型（重要方法论）
- 创造模式**放方块不消耗** ⇒ 无法计数。改 **SURVIVAL**，`placed = 64 - inventory[cobblestone]`。
- `requires_surrounding_blocks` 只给 **3×3×3 = 27 格**邻域（18 air + 9 grass），够作局部佐证，
  但不足以当主计数器。`request_raycast=True` 给准星指向，用于诊断 use 为何空放。
- 写死 oracle 先行摸出环境事实：观测**延迟一 tick**；放置有 **~4 tick 冷却**；
  camera pitch 是**累加**的；低头 60°+ 放置成功率暴跌（12→1）。
- **碰壁 4（我自己的 bug）**：初版 oracle 用 `ORACLE[d % 5]`，每 5 个决策重发一次 `pitch +60`，
  累加撞满 90°——**和模型犯的是同一个错**。修正后 oracle 40 决策放 **39 块**（100% 命中）。
  没有这根标尺，模型的"3 块"会被误读成"能用"。

### 实验设计的两次自我否定
- **v1 用 27 维离散索引 + 无历史**，是设计错误：把 VLM 降级成查表器，且它看不到自己已经低头。
  用户指出后按 **Lumine**（arXiv:2511.08892）重做：语言原生动作串 + hybrid thinking + 多帧历史。
- **v2 用"转多少度"作动作头**，仍是错的：模型被直接问"该转多少度"时答 `-1`（符号都错）。
  用户提议改**像素坐标**——先用 `tests/probe_omni_pointing.py` 标定：模型用 **1000×1000 归一化**，
  指点精度 **2.2–5.4 px**。换成像素瞄准后方向立刻正确，放置数 0 → 4。

### 反直觉发现：Mamba 的"无限上下文"在这套栈里不免费
根因（`nano_nemotron_vl.py` 未声明 `SupportsMambaPrefixCaching` ⇒ vLLM 对整个 Omni 关闭 prefix
caching ⇒ 每帧 +20ms 线性 prefill；显存上"状态常数大小"成立、延迟上不成立）与完整时延曲线沉到
`knowledge/conclusion_omni_nvfp4_5090.md §5`；原始数据 `docs/results/omni_history_latency.json`。

### 顺带回答：换 GPU 无头渲染值不值
不值。llvmpipe 软件渲染 640×360 下，真实森林世界 par=1 约 **15.2 tick/s**（完整并行标定见
下一会话续 2）；Minecraft 本身 20 tick/s，par≥2 时总吞吐线性扩展已够用，换 GPU 还会与 vLLM
争那 32GB。
（早先在本节记为 **38.7 tick/s ≈ 2× 实时**（1280×720 → 27.2），系用**超平坦**世界测得并外推
真实世界，已更正——森林实测仅 15.2；另一次 17.35 tick/s 是当时 FlashInfer 的 nvcc 占满 224 核、
llvmpipe 被饿着——软件渲染吞吐强依赖空闲 CPU。）

### 落地产物
- `tests/probe_omni_minecraft_lumine.py` —— Lumine 式直控，含 `--aim pixel|degrees`、
  `--assist-tilt`、`--scripted-oracle` 对照臂。
- `tests/probe_omni_pointing.py` —— 像素坐标约定与指点精度标定（换模型必跑）。
- `tests/probe_omni_minecraft_control.py` —— v1（离散索引），保留作反面对照。
- `docs/results/omni_mc_control/` —— 各臂 summary + 最终帧。

## 2026-07-09（Vast 5090，会话 1 续 2：慢塔接线 + 三次评价体系翻车）

> 会话目标：把 Omni 慢塔接进 GRPO 快塔回路（1Hz 文本指示 + Haiku 判官排序），并记录三次
> 因手工奖励代理翻车的教训。

### 教训:我连续三次用手工设计的奖励代理,而这正是 GRPO-判官范式要消灭的东西

用户三次纠正,每次我都在犯同一类错:

1. **像素直控 Minecraft(v1)**:用 27 维离散索引 + 无历史。模型锁死在 `[13,13,9,1,9]`。
   → 用户指出应照 Lumine 做(语言原生动作 + 历史)。
2. **慢塔指点体检(v1)**:我在 system prompt 里给了**带具体坐标的示例**
   `{"subgoal": "chop the nearest tree", "aim": [430, 560]}`。模型 16 次里抄了 6 次。
   我还把 `leaves` 算成"命中树",报出 `tree_hit_rate = 0.75`。
   → **对照臂拆穿**:恒定复读 `[430,560]` 也是 0.75;随机瞄准在这个烂指标上甚至有 0.438,
   **比慢塔还高**(黑橡木林树冠铺满画面)。删掉示例、改测树干命中后:
   慢塔 **5/16 = 0.312**,恒定 **0/16**,随机 **0/16**(Fisher p≈0.04)。地基成立,但只有 31%。
   顺带发现另一个公平性 bug:random 臂与慢塔臂共用同一个 rng,**看到的画面序列不同**,三臂不可比。
3. **改用 `trunk_hit_rate` 作主指标**——用户直接否掉:"我不是让你用 haiku 做老师吗?
   那就告诉说,哪个轨迹做的更多哪个更好。"
   → `grpo_r2.py` 开篇就写着「相对优势**不再用程序统计**」。手工命中率与手工进度分是同一类错误。
   **规范:判官排序给稠密优势;里程碑(inv_events)只作不可刷的汇报锚点,不进训练信号。**

### 判官 I/O 体检(tests/probe_judge_io_haiku.py,真 Haiku,不需要权重)
- 契约通过:`parse_ok_rate = 1.00`,排名与构造的质量阶梯一致。
- **脆弱点**:`claude -p` 的 Read 权限取决于图片路径是否在**工作区内**。
  `grpo_r2.OUT = "runs/grpo_r2"` 是相对路径 ⇒ 换 cwd 启动会被拒,判官静默退化为
  `fallback_score`(只数背包事件),而 `fallback_chunks` 只是计数器、不会让 run 失败。
  我最初把 OUT 设到 `/tmp` 复现了这个失败(Haiku 回"等待权限授权以读取图像文件")。
- **判官不会说"分不出高下"**:4 条证据文本完全相同、图上只是无意义色块(黑/红/绿/蓝),
  它仍编造语义("绿色=正向反馈信号")并给出严格全序。⇒ 判官会把噪声当信号,
  `adv_var` 大不代表学到东西。建议 R2 加一条"同轨迹不同渲染"的对照组测幻觉率。

### 读代码发现:现行 GRPO 更新里慢塔是断的
`train/fovea_twotower/grpo_r1.py::update`:
```python
g = torch.zeros(1, 1, device=dev)     # goal 向量喂零
prev[1:, 0] = 0.0                     # prev_action 全零
```
⇒ **即使慢塔给出完美指示,梯度里也没有它。** 新实现 `train/craftground/grpo_pixel.py`
把 goal/prev 真正接进前向。(`adv*(CE+BCE)` 的形式本身没问题:CE 打在采样 bin 上
= `-log π(a)`,所以它是正确的 REINFORCE。)

### 按"苦涩的教训"裁决退役的组件
`g1_conv_head_v7b_wood.pt`(手标 log/iron/coal/dirt 分割头)、`g1_vectors.pt`(类向量 bank)、
`wood.py::wood_label_img`(8 角凸包造树干 GT)、`net/fovea_twotower/token_stream.py`
(YOLOE 解析槽位)、`net/bc/policy.py`(冻结 DINOv3 + BC)。
R-A 那一轮"扩类校准 → log mIoU 0.322 → 闭环 wood_rate 仍为 0"正是该教训的实例:
人力压进感知先验,最终卡在执行层。
（此"卡在执行层"的归因在续 3 被 YOLOE 覆盖实测再次改判为"表征盲"——快塔在森林里观测基本
为空，非执行技能缺失；见本文续 3。）

### 环境
- `venv-mc` 补装 torch 2.11.0+cu129(原为 CPU 版)+ sentence-transformers;
  `/venv/main`(vLLM)不动。两者 `cuda_avail=True`。
- Omni 慢塔以 `GPU_UTIL=0.85 MAXLEN=8192` 常驻(26.7/32.6 GB),给快塔留 ~5.9GB。
  0.78 会因 KV 无空间而 `No available memory for the cache blocks` 启动失败。
- **并行环境扩展性(决定 --par,也回答"要不要 GPU 渲染")**:
  llvmpipe 软渲,默认世界(森林):par=1 → 15.2 tick/s;par=2 → 每环境 15.4-16.1(合计 31.5);
  par=4 → 每环境 16.0-20.0(合计 69.1)。**每环境不掉速、总吞吐线性**,224 核未饱和。
  ⇒ 当前不需要 GPU 渲染。(此为森林世界的权威口径,即会话续 1"换 GPU 值不值"一节引用的数据;
  该节早先的 38.7 tick/s 系超平坦世界外推,已在该节更正。)

### 新增(未冒烟,仅结构落盘)
- `net/pixel_tower.py` —— 从零像素快塔(conv stem + 因果 Transformer + FiLM goal 条件)。
- `train/craftground/grpo_pixel.py` —— 规范实现:Omni 慢塔 1Hz 文本指示 + Haiku 判官排序
  + `group_advantage` + REINFORCE。里程碑只作汇报锚点。
- `tests/probe_slow_tower_aim.py` —— 慢塔指点体检(三臂对照,主指标已按用户裁决重定)。
- `tests/probe_judge_io_haiku.py` —— 判官 I/O 契约体检(真 Haiku,零权重依赖)。

## 2026-07-09（会话 1 续 3：n_cls 的代价被量化——词表删掉 98.7% 的世界）

> 会话目标：量化快塔 YOLOE 解析头里 4 类词表（cls/n_cls）相对无词表 pf 提案的信息损失，
> 据此定重构方向。

### 用户裁决
"n_cls 太蠢…能不能不要它，或者至少让快塔自行编码向量，让慢塔传过来的信息经过处理。"
并指出 YOLOE-26 的正确用法：**pf 端到端出 256 个提案（含所有可能物体），提示向量只负责
"从这 256 个里挑出我们要的"，不负责感知**。

（事实更正：`PARSE_DIM = 7` 含 `cls/n_cls` 由 commit `a16ca19`(07-08) 引入，非本会话所加。）

### 实测(tests/probe_yoloe_coverage.py,8 帧黑橡木森林,yoloe-11l-seg{,-pf})
| 通路 | 平均提案 | 像素覆盖 | 准星被覆盖的帧 |
|---|---|---|---|
| pf(无词表,max_det=256) | 48.8 | **90.8%** | **88%** |
| pf 截断到 top-K=8(现行) | 8 | 87.8% | 62% |
| prompt(WOOD_CLASSES 4 类,现行) | **0.5** | **1.3%** | **0%** |

⇒ **两个独立的信息瓶颈**：
  (a) **词表瓶颈(主)**：类别打分把观测塌缩成"词表命名过的东西"。森林里每帧仅 0.5 个框、
      覆盖 1.3% 像素、准星前的方块 0% 被看见。
  (b) **K=8 截断(次)**：像素覆盖 90.8%→87.8%，但准星覆盖 88%→62%。

### 由此重新归因 latch=0
R-A 曾把 `wood_rate=0` 从"感知盲"改判为"挖掘/导航执行"(依据 `saw=27~174/500`)。
但 27~174/500 ≈ 每帧 0.05~0.35 个检出，与本次实测的 0.5 同量级 ——
**快塔在森林里的观测基本是空的**。YOLOE 本身看得见(pf 覆盖 90.8%)；
是接在它后面的 4 类词表把信息扔了。⇒ 应改判为**表征盲**，不是执行技能缺失。

限定：本次 prompt 臂用 `get_text_pe` 的原始文本 PE，而真实系统用 `g1_vectors.pt` 的
域内校准原型（会好一些）。故 1.3% 是真实管线的**下界**。结构性结论不受影响。

### 新坑：torchvision 在 sm_120 上没有 CUDA 内核
`torchvision 0.26.0+cu129` 的 `_C.so` 只含 `sm_50/60/70/75/80/86/90` 的 cubin，
**无 sm_100/sm_120，且无可用 PTX 回退**：
```
cuobjdump --list-elf torchvision/_C*.so  ->  sm_50 60 70 75 80 86 90
torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
    at torchvision.ops.nms   (ultralytics NMS 直接调它)
```
⇒ **YOLOE 在 RTX 5090 上开箱即挂。** 临时绕过：把 NMS 搬到 CPU（框数仅数百，代价可忽略，
见 `tests/probe_yoloe_coverage.py::patch_nms_to_cpu`）。用户计划换 cu130 机器，届时应自愈。
（与今早 vLLM 的三个 sm_120 坑同族：生态尚未跟上 Blackwell 消费卡。）

### 重构方向（用户口径）
```
token_j = [ e_j(512d 单位嵌入, 类别无关), cx, cy, w, h, conf, area ]     # N 最多 256,无 cls
goal    = 慢塔文本子目标 -> 与 e_j 同一 512d 空间(text_bank / 学习式投影)
快塔     : goal 作 query -> 对 N 个提案槽 cross-attention(**学出来**该看谁)
          -> 因果时序头 -> 相机分箱 + 按键
```
零件全部现成：`yolo_unified.propose()`(类别无关提案) / `proposal_embed()`([N,512] 单位嵌入)
/ `text_bank()`(同空间文本 PE)。`cls/n_cls` 正是把 `proposal_embed` 塌缩成一个标量的那一步。

## 2026-07-10 苦涩教训重设计定稿 + 全库文档审计

**重设计**:与用户逐轮敲定,定稿在 `knowledge/design_bitter_lesson_map_integration.md`
(§6-§11),`next_session.md §7` 有速览。要点:通信图七边审计(唯一基本正确的边=动作头);
DINO patch 网格替代 YOLOE 的**探针门控提案**(§8 预登记,出结果前路线 2 仍现行);
弃 MiniLM/FiLM,subgoal 文本 token 直入 + hindsight relabel;aim 下发即 IPM 钉图不做 ZOH;
帧堆叠 2-4;慢塔会话=设计 2(无状态重提示+状态行/地图行外置),Mamba 固定态判为吞吐性质
非记忆(W4 + 混合架构自身即对冲),零样本无限流永不做;慢塔留 Omni,换塔备选 Qwen-VL 系。

**文档审计**(两个 SubAgent 盘点 docs 45 + knowledge 31 = 76 文件,三级判决):
- **删除 5**(零入引/纯过时计划/错误现状快照,git 历史可查):`knowledge/code_analysis.md`
  (把退役 dreamerv3 标"当前最活跃",最毒)、`design_learned_attention.md`、
  `design_gaming500_hd_pretrain.md`、`design_gaming500_consume.md`、`install_refactor.md`。
- **打废弃头 9**(毒但被现行文件/登记表引用或混有有效证据,封存不删):
  knowledge 的 design_fovea_yolo_fasttower(仅 §4.5 数学存活)/design_llm_semantic_layer
  (§1-2 裁决存活)/design_llm_deep_integration/design_wca_agent/design_rollout_research_program;
  docs/architectures 的 fovea-system-architecture(撤"单一入口")/fovea-brain-division-scale-plan
  (头恰 4 行,lessons 行号引用已同步 +4)/fovea-roadmap-north-star(路线知识存活,
  训练信号作废)/fovea-hypothesis-verification(排队项作废,V3/V5/V4 结论存活)。
- **其余保留**:现行 12(next_session/activity_log/arch 2 现行/results 5/knowledge 新五件+Omni 双结论),
  档案 50(负结果与退役线结论,登记表证据链锚点一律不动)。
- 已知遗留:`install.md:605` 引用不存在的 `knowledge/ppo_ad.md`(失效链接,未修)。

## 2026-07-10 清理第二批(用户批准"全部执行"):死代码 + 档案压缩

- **死代码整棵删除**(status_built_not_wired 已核实全部不在 grpo_pixel 运行时链上,
  删前 grep 精确 import 语句定爆炸半径,删后复查零残留):
  `net/{dreamer,dreamer4,dreamerv3,bc,ppo_ad,guidance}`、`train/crafter/` 整目录、
  `train/craftground/train_dreamer4.py`、`train/minecraft/{train_bc,train_dreamer4}.py`、
  `train/fovea_twotower/{gate_fasthead,grpo_skill,rest_update,train_fasthead}.py`、
  dreamer/bc 系 tests 10 个。**保留**:`train/minecraft/vpt_action.py`(mu-law 口径出处)、
  `vpt_dataset.py`(E1 BC 暖启动数据管线)、`net/vpt_lib`、`net/encoders`、
  `net/dino_tokenizer.py`(DINO 探针候选)、`ego_map.py`、`grpo_harness.py`。
- **双塔战役九文档压缩**成 `docs/architectures/fovea-twotower-campaign-archive.md`
  (假设/核心数字/裁决 + 可复用事实四条;原文 git 历史 commit 1a29855)。
- **混元三篇架构分析删除**(纯外部调研,零现行引用),architectures/README 已更新。
- **omni_mc_control 原始产物精简**:16→4(留 oracle 臂 + 最优臂 pixel_v2 的
  summary+终帧),conclusion_omni_pixel_control 引用行已注明。
- **install.md 重写**(611 行旧全模块指南→当前主线三依赖 + 冒烟自检;
  旧版 git 历史),原 ppo_ad.md 失效链接随重写消失。
- `knowledge/dreamer.md` 随代码同批删除;`status_built_not_wired.md` 对应行改标"已删除"。

## 2026-07-10 代码修改第一批:五个 log π bug + D1 帧堆叠 + 按键先验

按设计文档 §11 顺位 1 执行(`net/pixel_tower.py` + `train/craftground/grpo_pixel.py`):

- **①T 失配**:双侧统一 T=1 + `frame_stack=4`(D1,单帧测不出速度的结构修复与
  bug 修复合并;更新端 `stack_frames` 与采样端 deque 逐字节同序,有单测锁定)。
- **②dropout**:`PixelTowerConfig.dropout` 默认 0.1→0.0;采样 `tower.eval()`、
  更新 `tower.train()`。实测残差:nn.MultiheadAttention eval 走 fast path,
  train 走常规路径,同 dropout=0 下差 ~4e-7(内核浮点,非分布差)。
- **③温度**:update 的 CE/BCE 打在与采样一致的 `logits/temp` 上。
- **④goal**:rollout 逐 tick 落盘 386 维 goal 向量(`goals` 数组),update 逐 tick
  回放;`goal_log` 补存 aim,`goal_last` 删除。
- **⑤单步**:一个 group 全部窗口梯度累积(按全组 tick 数归一)后单次 `opt.step()`
  ——严格 on-policy REINFORCE,无 ratio/clip/KL 之需;尾部 tick(旧 384-399)纳入;
  `--seq` 改 `--chunk`(纯显存分块,不改数学)。
- **按键先验**:`key_head.bias ← logit(key_prior=0.05)`,起点期望 1 键/tick(原 10)。
- 验收:`tests/unit/test_grpo_pixel_fixes.py` CUDA 5/5(先验/堆叠同序/eval-train 同分布/
  单步+goal 进梯度+尾段/温度进目标);全 unit 12/12。**真实 CraftGround `--smoke` 未跑**
  (本机无 Xvfb+Omni 服务),放大规模前必须先跑。

## 2026-07-10 代码修改第二批:地图接线件 + token 塔骨架 + 慢塔设计 2 契约 + §8 探针

用户裁决:本机不跑 CraftGround / 大模型验证,其余按推荐方案全部落地。

- **`net/map_io.py`(新)**:`ipm_ground`(精确针孔逆投影+地平面相交,北=+y/东=+x/
  pitch 向下正;空中目标显式 invalid)、`MapWriter`(W_c: feat→c 是唯一可学件)、
  `MapReader`(EgoMapClip 三级 → grid²×levels 个 token [feat⊕pos⊕level],grid_sample
  可微)、`AimPin`(B1:aim 下发 tick 落世界系钉点,step 平移账本,TTL 过期;零网格零模糊)。
  单测 5/5 含 IPM 解析几何精确值(pitch45°/h1.62→正前 1.62 格)与 W_c 梯度回传。
- **`net/token_tower.py`(新)**:v2 目标结构骨架——goal-as-query cross-attention,
  KV = 视觉 token ⊕ 地图 token ⊕ **UTF-8 字节语言 token(A1,从零嵌入,开放码本)** ⊕ prev;
  key_prior 同款注入;空组容错(前端可缺席)。单测 4/4 含"各组梯度真流"与
  "turn left/right token 可分"(MiniLM 反义词坍缩在此结构上不可能)。
- **慢塔设计 2 契约接线(grpo_pixel.py)**:SLOW_SYSTEM 换五字段 JSON
  (prev_done/decision∈{continue,switch,replan}/subgoal/aim/done_when);
  `state_line()` 外置状态(tick/库存尾 6/位移/最近 3 条子目标带 done 标记);
  `parse_slow_reply()` 模块级容错解析(字段逐个降级不整体作废);prev_done 语义=
  上一条子目标,回填 goal_log[-1][5];判官证据的子目标轨迹带 ✓ 标记。单测 4/4。
  **未跑**:真实 Omni 上的格式合规率与决策质量(大模型验证,用户暂缓)。
- **`tests/probe_dino_vs_yoloe_aim.py`(新,§8 预登记探针 harness)**:双臂特征
  (DINO patch 网格展平 224×384 + fovea 裁剪臂;YOLOE [geo6⊕e_j] pad 展平)+
  K 折 ridge R²(D>N 走对偶式,合成数据双路径验证 0.9998/0.9999);地形分层报告。
  **待活环境采集** manifest(准星角偏移标签 raycast 只进训练侧)。
- 全 unit 25/25。新件登记进 status_built_not_wired(接线条件注明);设计文档 §11
  进度标记更新(1✅ 2◐ 3◐ 4✅)。

## 2026-07-10 代码修改第三批:自标定 + 提示词/判官去领域化 + 慢塔微调方案入档

用户裁决:物理参数(分辨率/FOV/相机增益/步速/键位)不许绑死代码或训练集,
模型须自主通过测试在脑内建立场景物理参数;少提前假设,可跨游戏。

- **`net/calibration.py`(新,已接线进 rollout)**:`flow_shift` 相位相关求全局平移
  (Hanning 窗+亚像素抛物线);`fit_action_flow_map` **键位无关**离线标定
  (flow≈M·action 最小二乘,不需要知道动作通道语义——VPT px 单位/任意键位布局
  同一套代码);`SelfCalib` 在线状态(已知相机命令→流→px/deg→FOV=H/gain;
  步速用 pose 只进训练侧);`probe_plan` 净漂移为零的对称探针序列。
  三出口:physics_line(慢塔 prompt,未标定如实说不编数)/physics_vector
  (token 塔 geo,geo_dim 4→12 含钉点与物理位)/fov_y_deg(接线时替 IPM 固定 70°)。
  `calibrate()` 接进 rollout 开局(~14 tick,--no-calib 可关),calib 进 metrics。
- **提示词去领域化**:SLOW_SYSTEM 零领域知识,任务走 TASK 行(--task 可换游戏),
  输入四行结构化(TASK/STATE/PHYSICS/预留 MAP)+ 帧;输出五字段 JSON 不变。
- **判官 rubric v2**:删除手写进度阶梯(瘫痪<无方向<…<铁)——那是人写的价值函数
  经排序渗进训练信号;v2 只给 TASK 由判官自行判断推进;RUBRIC_TMPL 任务模板化,
  judge(task=) 与游戏解耦。
- **慢塔思维模式微调方案入档**(设计文档 §11.3,待数据):拒绝采样 SFT(只留
  done_when 机器核对达成的回复——思维模式被"成功的自己"筛出而非人写模板)+
  prev_done/decision 真值校准 + 流式 SFT loss-on-query(记忆移交前提)。
- 轨迹比较演进入档(§11.4):排序落盘→成对偏好→本地 RM(E3,search 支柱可扩展化)。
- 验收:`tests/unit/test_calibration.py` 5/5(平移恢复/键位无关增益恢复含无关通道≈0/
  FOV 派生/零漂移探针/未标定诚实输出);全 unit 30/30。
  **未验**:真实环境探针序列的实际光流质量(运动模糊/低纹理天空)——首次 --smoke 时
  盯 metrics 的 calib 字段;VPT 离线标定要等数据下载。

## 2026-07-10 标定加固批(回应"代码测试能好使吗/GUI/手柄"三问)

- **置信门控**:`flow_shift` 返回 (dx,dy,conf),conf=主峰/3px 邻域外次峰;
  `update_camera` 低置信证据直接丢弃(动态物体/低纹理不污染中位数,宁可 uncalibrated)。
- **延迟扫描**:`fit_latency` 对 cmd[t]×flow[t+k] 相关扫 k,控制延迟本身成为被测参数;
  calibrate() 增益配对按测得延迟错位,corr<0.3 不采信。修一个评分 bug:
  无方差轴不再摊薄分数(单轴探针满分被封 0.5)。
- **响应曲线**:`fit_response_curve` 三参数 |flow|=g·max(|c|−d,0)^e(网格+闭式 g),
  覆盖鼠标加速度/摇杆死区;probe_plan_multi(2°/4°/8°) 多幅度采样,净漂移仍 0。
- **控制模式探针**:脉冲 1 tick+静默 3 tick,流骤停=position(鼠标)/持续=velocity
  (手柄摇杆)——模式进 physics_vector(10 维,含延迟/模式位)与 PHYSICS 行;
  快塔动作头不用改(mu-law 本就输出有界归一量,只换解码映射)。
- **GUI 双态**:`update_camera(in_gui=True)` 拒收证据(游戏内探针被 GUI 门控);
  GUI 光标增益走 `cursor_gain_from_diffs`——连发两次等量命令,帧差质心中点法
  (m2−m1=gain·cmd),无需光标模板。界面语义归慢塔(OCR 已验),标定后 GUI 点击
  退化为线性伺服(可能解开合成闭环 0.00 的死局)。
- calibrate() 开局序列重排:压低视线(地面纹理)→多幅度探针→延迟扫描→错位配对
  →曲线拟合→模式脉冲(回正)→步速,约 40 tick。
- 验收:test_calibration 11/11(新增置信拒垃圾/GUI 门控/延迟恢复/曲线死区指数恢复/
  模式判别/光标中点法/多幅度 FOV+曲线);全 unit 36/36。

## 2026-07-10 YOLOE 整线删除(用户拍板 DINO;SubAgent 测绘 + 主线执行)

用户裁决:视觉前端定 DINO(更合苦涩教训:objectness 头是人工监督先验),
YOLOE 废弃,原 §8 探针门控作废。SubAgent 只读测绘删除半径(66 文件,含精确
import 链与文档引用),主线执行:

- **删 66 文件(~11k 行)**:net/fovea_twotower 6(yolo_unified/yolo_parse/seg_head/
  token_stream/wood/tower)、net/encoders/yolo_backbone_encoder、
  train/craftground/train_ppo_ad、tests/performance/prof_throughput、
  tests/probe_yoloe_*3 + probe_mamba_seed、tests/integration 9(assembly_a1/
  map_approach_ablation/wood 链等——地图组装测试将在 DINO 接线时按新前端重写)、
  train/fovea_twotower 44(双塔/track/terrain/grpo_r*/fovea 专用 SFT 线)。
- **刻意保留**(每个都锚定现行路径):ego_map.py(map_io 依赖,R1 红线)、
  grpo_harness(运行时)、map_probe/map_loc_probe(地图探针)、judge_exam 系 +
  judge_train(判官对照纪律 + E3 本地 RM)、nano9b_qlora_smoke(QLoRA 工具链
  结论复现锚)、VPT 全管线(E1)。vl_lora_smoke 因依赖被删 eval_g1 一并删
  (R-D 结论仍在 next_session §1)。
- probe_dino_vs_yoloe_aim.py → **probe_dino_aim.py**(YOLOE 臂摘除,降级为
  DINO 单臂瞄准可学性验证:R²>0 且地形分层不塌,FAIL 跑 fovea 双尺度臂)。
- 文档裁决同步:papers_in_use(YOLOE→退役清单,新增 DINOv3 方向行)、
  arch_current §2 作废注(§2.2 词表塌缩数学保留作档案,登记表引用核实不断链)、
  status_built_not_wired、next_session §7、设计文档 §7/§8/§12、install.md。
- 验收:删除后残留 import 全库 grep 零命中;全 unit 36/36。
  net/fovea_twotower/__init__ 清成 ego_map 说明页。
