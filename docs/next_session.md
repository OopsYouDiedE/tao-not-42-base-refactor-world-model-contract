# 交接单（2026-07-10 重写，给下一个训练会话——可能换模型、换机器）

> **使命：GRPO-Pixel 主线已完成链路验证（五个 log π bug 已修 + CraftGround `--smoke`
> 全链路通过 + VPT BC 暖启动第一批入档）。下一步按 §2 待办推进：DINO 前端接线、
> BC 数据扩容、带慢塔机器上的 GRPO 精修。**
>
> 本单于 2026-07-10 重写为单一现行交接单：只保留现行结论（§1）、待办（§2）、
> 已修必修项的验收记录（§3）与纪律（§6）。此前叠加的"已作废的设定"整节、
> 旧设定逐行裁决表、路线待决点（已由用户裁决关闭）均已删除，git 历史可查。
> 定稿设计的唯一入口：`knowledge/design_bitter_lesson_map_integration.md`（§6–§12）。

---

## 1. 现行结论（直接当结论用，勿重跑）

- **心跳延迟**：85ms@1.5B / 94ms@2B-VL（448×256）。慢塔硬件预算基线。
- **Omni NVFP4 慢塔在单卡 5090 原生可用**（2026-07-09 实测，`conclusion_omni_nvfp4_5090.md`）：
  权重 21.5GiB、图像 TTFT 0.154s（热）、ASR WER 0、Crafter 帧语义与像素指点（2.2–5.4px）全部合格。
  四个 sm_120 问题的修复已内联进 `tests/serve_omni_nvfp4.sh`。**结论：感知/指点质量无损，
  可自托管当 episode 级慢塔；但零样本不能当控制器**（`conclusion_omni_pixel_control.md`：
  像素直控最好 4 块 vs 手写 39 块；病灶是"感知→动作不接"，相机连续标定符号/增益 LLM 无法完成）。
- **I_gui**：16×16 灰度 + 逻辑回归判定 GUI 开合，准确率 99.91%。
- **IPM 数学精确**（单测含解析几何精确值）。
- **chunk_k=1 裁决**（R-B）：离线首步 cam_acc 随 k 单调降（k1 0.275 / k4 0.270 / k8 0.256，k8 破 5% 门）、
  闭环 switch 随 k 崩（k1 0.125 > k4 0.125 > k8 0.000）；块内均匀权重。已固化进 `PixelTowerConfig.chunk_k=1`。
- **慢塔 LoRA 配方**（R-C 1.5B + R-D VL）：r16 / q,k,v,o / lr1e-4（reason_delta 配方，新 adapter，防复核回退）；
  VL 视觉塔冻结。**注意：训练器代码（heartbeat_sft/reason_delta_sft）已于 prune3 物理删除，
  执行 §2-4 需要慢塔微调时按本配方重建。**
- **VL LoRA 冒烟配置模板**（R-D，PASS）：448×256，batch1×梯度累积8，grad-ckpt on，r16 qkvo；
  loss 降 66.6%，峰值显存 5.06GB，无 NaN。
- **状态行 schema / 微决策词表 / 僵局阈值**（R-C）：
  `t=<tick> 库存:… 可见:<cls(dist格)> 位移:<m>m goal:<cls>`；词表 {继续, 换目标:<cls>, 重规划}；
  **N=20**（sweep 10/20/40 → 留出决策 acc 0.349/0.937/0.841）；留出决策 acc 0.937 / 格式合规 1.0。
- **9B QLoRA 本地工具链**（R-F，PASS）：NF4 + LoRA（qkvo）反向跑通，loss 3.8→0.04，峰值 13.38GB。
  投影不能压 4bit 的结论成立；精度档位以实测为准：官方 NVFP4 里 Mamba `in_proj`/`out_proj` 是 **FP8**，
  只有 `A_log`/`D`/`conv1d`/`dt_bias` 留 **BF16**（见 `conclusion_omni_nvfp4_5090.md §2`）。
  NF4 硬压 Mamba 投影会触发 mamba-ssm 融合内核的 raw `F.linear` 报错，
  `llm_int8_skip_modules=[out_proj,conv1d,lm_head]`。
- **稀疏键（E/背包/合成等）归慢塔**：视觉平均 BC 学不到稀疏+精准+零容差的关键动作
  （`conclusion_fasttower_skill_ceiling.md`：合成闭环 0.00，BC holdout loss 0.0013 但只学到背景）。
- **GRPO 起效的受控对照**（`conclusion_fasttower_skill_ceiling.md`，**决定 GRPO-Pixel 成败的核心先验**）：
  GRPO 能否起效由"目标是否视觉可见（能否产生奖励信号）"决定——可见目标（木头）0.50→0.81，
  不可见目标（石头贴石头墙）0.31→0.06 退化；且两次都是从 **BC 暖启动**起跑。
  aim+attack+导航族从视觉 BC 全学得会（0.31–0.69）；合成 GUI 学不会（动作模态问题，非对齐问题）。
- **渲染选型**（`conclusion_craftground_run.md §3/§3.1b`，L4 实测，复跑 `tests/bench_render_craftground.py`）：
  端到端、无争抢口径下 **ZEROCOPY 最快**（107.2 sps，比 Xorg+RAW 快 41%，python 侧 CPU 低 8×）；
  采集与训练同卡并行时选 Xorg+RAW（37.3 vs 25.3 sps）。ZEROCOPY 必须先经
  `patch_craftground_native()`（craftground 2.6.15 两处上游 bug 的运行时 shim）。
  EGL 无 X 渲染：驱动层可用，CraftGround 栈不支持（缺口在 Minecraft/GLFW 窗口层，需上游特性）。
- **VPT BC 暖启动第一批**（2026-07-10，`conclusion_fasttower_skill_ceiling.md` 末节）：
  holdout ce+bce **0.6350@run3-600**，cam_acc 0.830 vs 基线 0.8225，keyF1 0.651；
  checkpoint `runs/checkpoints/bc_vpt/best.pt`，`grpo_pixel.py --init-from` 就绪。
  过拟合点恒在 600–1200 步 ⇒ 加数据是纯规模杠杆（公开 blob 有数千段可继续加）。
- **CraftGround `--smoke` 全链路通过**（2026-07-10，L4）：判官真排序（fallback=false）、
  自标定首次真实环境实测（cam_gain 1.09px/deg / fov_y 79° / latency 1 tick / speed 0.174blk/tick）、
  BC checkpoint 严格加载；慢塔按设计降级（L4 无法跑 NVFP4）。链路验证完成。

## 2. 待办（顺序即优先级）

1. **DINO 前端接线——已完成（2026-07-10 后半）**：`grpo_pixel --tower v2` 接入
   TokenPolicyTower（DINOv3 patch 60 token/帧×S=2 + EgoMapClip 地图 48 token +
   subgoal UTF-8 字节语言 token，goal-as-query），装配模块
   `train/craftground/tower_v2.py`；默认仍 v1，checkpoint 分文件，bc_vpt 兼容有单测
   （`tests/unit/test_tower_v2.py` 7 项）。yaw/pitch 符号标定已做：部署侧取光流增益
   符号（`SelfCalib.yaw_sign/pitch_sign`，几何普适），训练侧 env-pose 角映射由探针
   采集器 `fit_angle_map` 实测（数值 `runs/probe_aim/pose_calib.json`，§2-2）。
   **v2 `--smoke` 链路验收 PASS**（2026-07-10，L4，真慢塔 Qwen3-VL-8B-FP8 +
   Haiku 判官）：判官真排序 fallback=false、slow_fail=0、自标定全实测
   （cam_gain 1.08px/deg / fov_y 79° / latency 1tick / speed 0.174blk/tick）、
   更新执行、checkpoint 落 `runs/grpo_pixel/tower_v2.pt`（v1 的 tower.pt 不受影响）。
   遗留：v2 的 BC 暖启动未接（GRPO 更新回放记录 token，MapWriter.w_c 梯度需 BC 侧
   同图重放）；relocalize 周期修正与慢塔 MAP 行未接。明细
   `knowledge/status_built_not_wired.md`。
2. **DINO 瞄准可学性探针——已跑，判决 PASS（2026-07-10 后半）**：活环境采集
   104 样本（24 episodes、5 个产样 seed、树干目标 raycast 标签，只进训练侧），
   DINOv3 冻结 patch + ridge 5 折：**R²_all=0.899，hole 0.885 / slope 0.881，
   地形分层不塌 ⇒ 按预登记判据 PASS**，fovea 双尺度臂不触发。
   两条如实限制：(a) 留 seed 折 R²=0.241（仅 5 场景，跨场景功效低，仍>0）——
   场景级泛化结论需更多 seed；(b) flat 层空缺（heightmap 地形分类把树冠高度算进
   起伏，林地样本全落 hole/slope，分类器工件非地形事实）。
   位姿符号标定实测（`runs/probe_aim/pose_calib.json`，fit_angle_map）：
   cmd→env 增益 g_yaw=0.975、g_pitch=0.99（约 1:1,符号正）；env_yaw =
   atan2(east,north)+180°（sign=+1,offset≈±180°,resid 1.7–4.7°）;
   env_pitch = down 角（sign=+1,offset≈-3°）。部署侧符号(光流增益符号)与此一致。
3. **BC 数据扩容 + hindsight relabel**：数据量是当前纯规模杠杆（§1 末节）；
   hindsight relabel 语言标注（A1 语言通道 grounding 的数据来源）未做。
   ▶ 2026-07-10 已启动扩池实测：下载器循环 8xx/9xx/10xx 索引（目标 +360 段），
   `bc_vpt2` fresh 训练 60k 步与 Qwen 慢塔共卡，产物 `runs/checkpoints/bc_vpt2/`。
   **gaming500 用户裁决（2026-07-10）**：多样化数据定位为第二阶段预训练语料
   （预训练→微调两段式，相机按逐录像统计量归一），不混入第一阶段 BC。
   触发条件：VPT 同域数据到千段量级且 holdout 曲线平台化、模型已加大之后。
   依据：Dreamer4 数据变量实验中 gaming500 MC 子集同预算差于 VPT；小模型容量
   被异域数据稀释。IDM 反标网络视频撤销（训练成本超预算），不再列为选项。
   BC 分布不均的三个手段按序消融（等扩池 run 结果后做）：相机两级头
   （动/不动 × 非零档 CE，契约改动需接线）→ 稀有键事件加权窗口采样
   （扩展 vpt_dataset.motion_sample 机制）→ 稀疏键 pos_weight 封顶 5–10 小消融。
   稀疏键可学性上限已有受控结论（视觉平均 BC 学不会，归慢塔），加权不改变该结论。
4. **GRPO 精修**：前置是有慢塔的机器（5090/Blackwell）或换塔（备选见设计文档 §10）。
   每 run 必须带判官退化输入对照臂（§6 纪律）。
5. **候选项（不阻塞）**：craftground 上游若修掉 ZEROCOPY 每帧 register/unregister，
   env.step 的 4.3 ms 差距还会缩小；CraftGround 全无 X 渲染需上游 mixin/GLFW 特性，
   估数天级且收益存疑（Xorg 路径已通）。
6. **待用户复核的清理边界项**（prune3，清单 `runs/prune3_manifest.md`）：
   - Godot 线：`train/godot_meta_rl/vec_env.py` + `utils/godot_rl/*` 是 assets 禁触子系统的
     唯一驱动桥。**用户 2026-07-10 裁决：保留（未来可能启用），本复核项关闭。**
   - `train/craftground/action_contract.py:22` 指向已删 vpt_lib 的注释：该文件属硬禁触区未改，
     是无害历史注释（口径已内联），禁触解除后再处理。

## 3. 五个 log π 必修 bug——已修（2026-07-10），验收记录

采样端与更新端曾在五处使 `log π(a)` 算错；数学论证与逐条修法的完整档案见
`knowledge/arch_current.md §6`。修复内容：

1. **T 失配**：双侧统一 T=1 + 帧堆叠 S=4（同时消灭速度盲，D1）；
2. **dropout**：采样 `eval()` / 更新 `train()`，`dropout` 默认改 0；
3. **温度**：损失与采样打在同一套 `logits/temp` 上；
4. **goal 回放**：goal 逐 tick 落盘 386 维向量并回放，`goal_log` 补存 aim；
5. **单步更新**：组内梯度累积后单次 `opt.step()` = 严格 on-policy REINFORCE，尾部 tick 纳入。

另注入按键先验 `key_head.bias = logit(0.05)`。
数学验收：`tests/unit/test_grpo_pixel_fixes.py`（CUDA 5/5 通过）。

## 4. 预算事实（当前是冒烟规模，不是训练规模）

默认 8 组 × 4 条 × 400 tick = 12800 环境步 ≈ 10.6 分钟游戏时间；**判官仅被调用 8 次**（每组 1 次）。
`--smoke` 进一步压到 groups=1 / ticks=120。这套默认值只够验链路，不足以产生可信的学习信号。
判官带宽：提高组数与每组条数比堆 tick 更划算（信息 ∝ log(n!)·组数，与 tick 数无关）。

## 5. 定稿设计入口

当前运行时 = `train/craftground/grpo_pixel.py` + `bc_vpt_warmstart.py`。
定稿设计 = `knowledge/design_bitter_lesson_map_integration.md`（§6–§12），其 §12 是执行清单；
建成未接线部件的判据与 file:line 明细 = `knowledge/status_built_not_wired.md`。
细节勿凭记忆，读设计文档。

## 6. 纪律提醒（违者重来）

**沿用**（与用户当面定过）：

- 教师必须是学生观测的函数；教师必须确定性；特权信息只进训练侧。
- 每级升级要有上一级在给定预算内的证伪记录，不许跳级。
- 判官调用全量落盘（蒸馏本地判官的数据在攒）。
- 运行时零脚本：宏只活在采集器；闩锁不进部署回路。
- 结论覆盖写进既有定论文件，不新建文件；物理参数不绑死代码或训练集（自标定，
  测不出就置 None，`net/calibration.py` 立场）。

**新增两条（从 2026-07-09 三次评价体系失误提炼）**：

- **禁止手工奖励代理；任何手工统计量当训练信号或主指标前，必须先被对照臂证伪。**
  三次失误同属一类错：27 维离散索引查表、system prompt 里放带坐标的示例导致 `tree_hit_rate=0.75`
  （恒定复读臂同样得到 0.75，随机臂在该失真指标上甚至 0.438）、以及用 `trunk_hit_rate` 当主指标。
  GRPO-判官范式的立意是"相对优势由判官排序给，不由手工程序统计给"；手工命中率与手工进度分是同一类错。
  里程碑（`inv_events`）只作**不可刷的汇报锚点**，不进训练信号。对照臂之间还须共享同一环境种子/画面序列——
  random 臂与慢塔臂曾共用同一 rng 却看到不同画面序列，三臂根本不可比。

- **判官会把纯噪声排成严格全序；`adv_var>0` 不等于学到东西。**
  实测（探针已删，结论入档 `knowledge/lessons_do_not_retry.md`）：4 条证据文本完全相同、
  图上只是无意义色块，判官仍编造语义并给出严格全序，且从不主动说"分不出高下"。
  故每个 judge-driven run 必须带一组"同轨迹不同渲染 / 退化输入"对照，测判官幻觉率。
  判官读图依赖 `claude -p` 的 Read 权限，图片路径必须在工作区内（`grpo_pixel.py` 的
  `OUT=runs/grpo_pixel` 是相对路径，换 cwd 启动会被拒 → 静默退化为 `fallback_milestone`
  机器分，只数背包事件，不会让 run 失败——必须监控 `metrics.jsonl` 的 `fallback` 字段）。
