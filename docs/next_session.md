# 交接单(2026-07-09 重写,给下一个训练会话——可能换模型、换机器)

> **使命:先把 `train/craftground/grpo_pixel.py` 的五个必修 bug 修到"实验结果可解释",
> 再让用户在快塔输入契约(路线 1 从零像素 vs 路线 2 类别无关提案 token)之间拍板,
> 然后才谈训练规模。当前 GRPO-Pixel 链路只跑通过冒烟,没有一次可解释的训练。**
>
> 背景:本单前身是 2026-07-08"今晚定标作战单"(把 fovea 双塔 / 分块 BC / 扩类校准的大规模设定钉死)。
> 2026-07-09 用户裁决按"苦涩的教训"退役了整条感知先验路线,该作战单里有若干行随组件失效——
> 见下方"已作废的设定",逐行标注,未静默删除。旧版本 git 可查。

---

## 1. 已验完毕、直接当结论用(勿重跑)

以下结论与被退役的感知路线无关,仍然成立:

- **心跳延迟**:85ms@1.5B / 94ms@2B-VL(448×256)。慢塔硬件预算基线。
- **Omni NVFP4 慢塔在单卡 5090 原生可用**(2026-07-09 实测,`conclusion_omni_nvfp4_5090.md`):
  权重 21.5GiB、图像 TTFT 0.154s(热)、ASR WER 0、Crafter 帧语义与像素指点(2.2–5.4px)全部合格。
  四个 sm_120 坑已内联进 `tests/serve_omni_nvfp4.sh`。**结论:感知/指点质量无损,可自托管当 episode 级慢塔;
  但零样本不能当控制器**(`conclusion_omni_pixel_control.md`:像素直控最好 4 块 vs 手写 39 块;
  病灶是"感知→动作不接",相机连续标定符号/增益 LLM 拿不下来)。
- **I_gui**:16×16 灰度 + 逻辑回归判定 GUI 开合,准确率 99.91%。
- **IPM 数学精确**。
- **chunk_k=1 裁决**(R-B):离线首步 cam_acc 随 k 单调降(k1 0.275 / k4 0.270 / k8 0.256,k8 破 5% 门)、
  闭环 switch 随 k 崩(k1 0.125 > k4 0.125 > k8 0.000);块内均匀权重。已写死进 `PixelTowerConfig.chunk_k=1`。
- **慢塔 LoRA 配方**(R-C 1.5B + R-D VL):r16 / q,k,v,o / lr1e-4(reason_delta 配方,新 adapter,防复核回退);
  VL 视觉塔冻结。
- **VL LoRA 冒烟配置模板**(R-D,PASS):448×256,batch1×梯度累积8,grad-ckpt on,r16 qkvo;
  loss 降 66.6%,峰值显存 5.06GB,无 NaN。
- **状态行 schema / 微决策词表 / 僵局阈值**(R-C):
  `t=<tick> 库存:… 可见:<cls(dist格)> 位移:<m>m goal:<cls>`;词表 {继续, 换目标:<cls>, 重规划};
  **N=20**(sweep 10/20/40 → 留出决策 acc 0.349/0.937/0.841);留出决策 acc 0.937 / 格式合规 1.0。
- **9B QLoRA 本地工具链**(R-F,PASS):NF4 + LoRA(qkvo)反向跑通,loss 3.8→0.04,峰值 13.38GB。
  **投影不能压 4bit 的结论成立,但精度档位更正**:官方 NVFP4 里 Mamba `in_proj`/`out_proj` 是 **FP8**,
  只有 `A_log`/`D`/`conv1d`/`dt_bias` 留 **BF16**(旧单写成投影留 BF16,记错;见 `conclusion_omni_nvfp4_5090.md §2`)。
  NF4 硬压 Mamba 投影会撞 mamba-ssm 融合内核的 raw `F.linear`,`llm_int8_skip_modules=[out_proj,conv1d,lm_head]`。
- **稀疏键(E/背包/合成等)归慢塔**:视觉平均 BC 学不到稀疏+精准+零容差的关键动作
  (`conclusion_fasttower_skill_ceiling.md`:合成闭环 0.00,BC holdout loss 0.0013 却背景背下来)。
- **GRPO 起效的受控对照**(`conclusion_fasttower_skill_ceiling.md`,**决定 GRPO-Pixel 成败的核心先验**):
  GRPO 能否起效由"目标是否视觉可见(能否产生奖励信号)"闸门决定——可见目标(木头)0.50→0.81,
  不可见目标(石头贴石头墙)0.31→0.06 退化;且两次都是从 **BC 暖启动**起跑。
  aim+attack+导航族从视觉 BC 全学得会(0.31–0.69);合成 GUI 学不会(动作模态问题,非对齐问题)。

---

## 2. 已作废的设定(因 2026-07-09 组件退役而失效,逐行标注,勿沿用)

**退役组件**(用户裁决,"苦涩的教训"):`g1_conv_head_*` 分割头、`g1_vectors.pt` 类向量 bank、
`net/fovea_twotower/wood.py::wood_label_img`(8 角凸包树干 GT)、`net/fovea_twotower/token_stream.py`
(YOLOE 解析槽位)、`net/bc/policy.py`(冻结 DINOv3 + BC)。

旧"已验完毕"里的一条随之失效:

- **"校准向量与文本 PE 基近正交,适配器须换基,blocked on 类对数"** —— **作废**。
  它依赖类向量 bank + 类对数,这条通路已退役。

旧"大规模训练设定总表"逐行裁决:

| 旧行 | 状态 | 原因 |
|---|---|---|
| 快塔 chunk k / 块内权重 = k=1 | **有效** | 与感知路线无关,已固化进 `PixelTowerConfig.chunk_k=1` |
| 快塔规模×示范量预算 ≤22M@83局 | **作废/需重定** | 该数字是在退役的 fovea BC 塔(冻结 DINO + fovea)上外推的,不迁移到从零 conv 的 PixelTower;唯一存活的是定性结论"主杠杆=移动天花板(GRPO/更好教师/数据),非买容量" |
| BC 配方(帽3×/prev_drop0.5/switch_os0.5/d512·L7)= 沿 v17 | **悬置** | 是 TrackNavTower(track_cmd)BC 配方;GRPO-Pixel 当前从随机初始化起跑、无 BC。是否复用取决于 §5 路线裁决与是否补 BC 暖启动 |
| **扩类校准配方(数据配比/neg_frac/闸门)= v4铁班+wood_gt+wood_negcert;neg_frac0.35;20ep;ncls=5** | **作废** | 整行依赖退役的 `g1_conv_head`/`wood_gt`/`wood_negcert`/`wood_label_img`/`ncls`。且归因已更正:R-A 曾把 `wood_rate=0` 判为"挖掘/导航执行技能缺失",现更正为**表征盲**——4 类词表把 YOLOE 每帧 48.8 个提案 / 90.8% 像素覆盖压到 0.5 个框 / 1.3% 覆盖 / 准星 0% 覆盖(`tests/probe_yoloe_coverage.py`)。词表本身就是被退役的领域先验 |
| 状态行 schema / 微决策词表 / N=20 | **有效** | 见 §1 |
| 慢塔 LoRA(r/targets/lr) | **有效** | 见 §1 |
| VL 图像分辨率/批量/grad-ckpt | **有效** | 见 §1 |
| GUI GT 工厂配方 / 掩码表(R-E) | **未跑/待重定** | R-E 从未执行;GUI 路线的 GT 工厂若日后重启需另立配方 |
| 混合架构工具链(本地侧,R-F) | **有效(档位已更正)** | 见 §1 的 FP8/BF16 更正 |
| 判官设定(4条/组排名制,全量落盘) | **有效,但新增两条约束** | 见 §6 新增纪律 |

---

## 3. 必修项——阻塞一切训练(在 `train/craftground/grpo_pixel.py`)

> ✅ **2026-07-10 已修**:五条全部落地(①改为双侧 T=1+帧堆叠 S=4,失配从结构上消灭;
> ②eval 采样/train 更新且 dropout=0;③损失打在 logits/temp 上;④goal 逐 tick 落盘
> 386 维向量并回放,goal_log 补存 aim;⑤组内梯度累积单次 opt.step=严格 on-policy
> REINFORCE)。另注入按键先验 bias=logit(0.05)、尾部 tick 不再丢弃。
> 数学验收:`tests/unit/test_grpo_pixel_fixes.py`(CUDA 5/5 过)。
> **未验**:真实 CraftGround 冒烟(需 Xvfb+Omni 服务)——放大规模前先跑一次 `--smoke`。
> 以下原文保留作论证与验收判据。

以下 1/2/3 三条各自独立地让 `log π(a)` 算错;不修则任何实验结果都不可解释。
主会话已核实,直接照修,勿重新论证。

1. **训练/推理分布失配**。采样 `grpo_pixel.py:236` 喂 T=1(快塔实际是无历史单帧策略,位置编码恒取 `pos[:, :1]`);
   更新 `grpo_pixel.py:286` 喂 T=64 因果序列。
   *修法*:采样端与更新端用同一时序长度(要么都单帧、要么都带历史窗口)。
   *不修的后果*:被求梯度的分布 ≠ 采样分布,REINFORCE 的 `log π(a)` 打在另一个模型上,优势方向失去意义。

2. **采样时 dropout 未关**。`grpo_pixel.py:333` 建模后全程无 `.eval()`,而 `PixelTowerConfig.dropout=0.1`;
   `torch.no_grad()`(`:238`)不关 dropout。
   *修法*:采样前 `tower.eval()`,更新前 `tower.train()`。
   *不修的后果*:采样时的策略是被随机置零扰动过的策略,记录下的 `log π(a)` 与真实策略不符。

3. **温度不一致**。`grpo_pixel.py:240-241` 用 `logits/temp` 采样,`grpo_pixel.py:291-292` 的 CE/BCE 打在
   **未除温度**的 logits 上。
   *修法*:采样与损失对同一套(除温度或不除温度)logits 求。
   *不修的后果*:采样分布与被优化分布温度不同,`log π(a)` 系统性偏移。

4. **goal 在梯度里被换成最后一个**。rollout 每 20 tick 刷新 goal(`goal_log` 已记录逐步值),但
   `grpo_pixel.py:285` 用 `r["goal_last"]` 重放整条 400 步轨迹。
   *修法*:更新时按 tick 回放对应时刻的 goal(用 `goal_log` 重建逐步 goal 序列)。
   *不修的后果*:慢塔指示在梯度里被抹成常量,前 380 步的条件信息进不了梯度,慢塔等于没接。

5. **组内轨迹被顺序更新**。`grpo_pixel.py:284-297` 每个 64 帧窗口一次 `opt.step()`,一个 group ~24 次更新
   复用同一批 adv,且**无 importance ratio / 无 clip / 无 KL to ref**——只有"组内优势"一件。
   *修法*:一个 group 的所有窗口梯度累积后单次 `opt.step()`;补 importance ratio + clip(+ KL to ref)使之成为完整 GRPO,否则明确标注为 REINFORCE 并接受其方差。
   *不修的后果*:后面的窗口用的是已被前面窗口改动过的策略与过期的 adv,这不是 GRPO,收敛行为不可解释。

### 设计缺陷(非 bug,但决定成败)

- **20 键独立 Bernoulli,随机初始化下每 tick 平均按下约 10 个键**(背包反复开合、hotbar 全按)。
  M-IRON Step0 预登记的失败模式③"抽搐乱动"在此结构下是**必然**而非风险。
  *修法*:`key_head.bias` 初始化到 `logit(人类按键率)` ≈ −2.9(`net/pixel_tower.py:124` 的 `key_head`)。
- **`aim` 是空间信息却经 FiLM 全局调制**(`net/pixel_tower.py:130`),而 `_ConvStem` 已把空间结构 flatten 成
  256 维向量(`:92`)。要学的恰是"目标像素 (x,y) → 相机增量",该结构对此映射最不友好(无平移等变性)。
  *方向*:让 aim 以保空间的方式进网络(在 flatten 前与特征图对齐,或走 §5 路线 2 的 goal-as-query cross-attention);
  与 §5 裁决联动,不单独动手。

### 一条被自己实证结论违背的做法(必须与用户对齐)

`conclusion_fasttower_skill_ceiling.md` 的受控对照显示:GRPO 两次起效都是从 **BC 暖启动**起跑
(可见目标 0.50→0.81);当前 `grpo_pixel.py` 从**随机初始化**起跑,去掉了唯一被验证有效的前提。
从随机初始化 + 判官稀疏优势(每组仅 1 次判官调用,见 §4 预算)直接 GRPO,与已验证的成功配方相悖。
**这不是助手能自行决定改不改的——上报用户:是否先补 BC 暖启动再 GRPO。**

---

## 4. 预算事实(当前是冒烟规模,不是训练规模)

默认 8 组 × 4 条 × 400 tick = 12800 环境步 ≈ 10.6 分钟游戏时间;**判官仅被调用 8 次**(每组 1 次)。
`--smoke` 进一步压到 groups=1 / ticks=120 / seq=32。这套默认值只够验链路,不足以产生可信的学习信号。
放大规模前,§3 五个 bug 必须先修,否则放大的是不可解释的噪声。

---

## 5. 待决裁决点:快塔输入契约走哪条(**待用户拍板,助手不得默认选择**)

两条路线在仓库里同时存在且互斥。**不得由助手替用户选。**

- **路线 1**:`net/pixel_tower.py` 的从零像素 conv stem(IMPALA 风格 4 层卷积 + FiLM goal 条件)。
  当前**唯一被 import 的实现**(`grpo_pixel.py:56`)。
- **路线 2**:commit `ff9b8c0` 的类别无关提案 token `[e_j(512d), cx, cy, w, h, conf, area]`
  + goal-as-query cross-attention(慢塔文本子目标投到同一 512d 空间当 query,对 N≤256 个提案槽做
  cross-attention,让快塔**学**出该看谁)。是 2026-07-09 更新的用户裁决(activity_log 续3)。

论据并列(不得遗漏,不得替用户加权):

- `conclusion_fasttower_skill_ceiling.md` 证明**冻结 DINO-CLS 的视觉 BC 能学会 aim+attack 全族**(0.31–0.69),
  GRPO 把砍木从 0.50 推到 0.81 → 支持"大规模预训练的通用表征"有用。
- **"苦涩的教训"反对的是人工领域先验**(词表 / 凸包 GT / 手标分割头),**而非大规模预训练的通用表征**
  (DINOv3、类别无关 YOLOE 提案器属于后者)。路线 2 用的是类别无关提案(通用表征),与该教训不冲突。
- 路线 1 从零 conv 抛弃了预训练感知,须**仅靠判官稀疏优势**从零学感知——这是 `conclusion_fasttower`
  里最难的设置(干净示范量都关键:同 iron 新采 24 局 0.688 vs 旧 500 步 0.25)。
- 路线 2 现实阻塞:YOLOE 在 sm_120(RTX 5090)上开箱即挂——`torchvision 0.26.0+cu129` 的 `_C.so` 无
  sm_100/sm_120 cubin 且无 PTX 回退,`ops.nms` 直接 `no kernel image`。临时绕过 = NMS 搬 CPU
  (`tests/probe_yoloe_coverage.py::patch_nms_to_cpu`);用户计划换 cu130 机器后应自愈。
- n_cls 实测(`tests/probe_yoloe_coverage.py`,黑橡木森林):pf 无词表 48.8 提案 / 90.8% 像素 / 88% 准星覆盖;
  4 类词表塌到 0.5 提案 / 1.3% 像素 / 0% 准星。**信息不是 YOLOE 丢的,是接在它后面的词表丢的**——
  这正是路线 2 去掉 `cls/n_cls`、保留类别无关 `proposal_embed` 的动机。

---

## 6. 纪律提醒(违者重来)

**沿用**(与用户当面定过):

- 教师必须是学生观测的函数;教师必须确定性;特权信息只进训练侧。
- 每级升级要有上一级在给定预算内的证伪记录,不许跳级。
- 判官调用全量落盘(蒸馏本地判官的数据在攒)。
- 运行时零脚本:宏只活在采集器;闩锁不进部署回路。
  (旧单里"R-A rollout worker 的 raycast 闩锁暂留"随该感知路线退役已作废,不再是活的脚手架。)

**新增两条(从 2026-07-09 三次评价体系翻车提炼)**:

- **禁止手工奖励代理;任何手工统计量当训练信号或主指标前,必须先被对照臂拆穿。**
  三次翻车同属一类错:27 维离散索引查表、system prompt 里塞带坐标的示例导致 `tree_hit_rate=0.75`
  (恒定复读臂同样刷到 0.75,随机臂在烂指标上甚至 0.438)、以及用 `trunk_hit_rate` 当主指标。
  GRPO-判官范式的立意就是"相对优势由判官排序给,不由手工程序统计给";手工命中率与手工进度分是同一类错。
  里程碑(`inv_events`)只作**不可刷的汇报锚点**,不进训练信号。对照臂之间还须共享同一环境种子/画面序列——
  random 臂与慢塔臂曾共用同一 rng 却看到不同画面序列,三臂根本不可比。

- **判官会把纯噪声排成严格全序;`adv_var>0` 不等于学到东西。**
  实测(`tests/probe_judge_io_haiku.py`):4 条证据文本完全相同、图上只是无意义色块(黑/红/绿/蓝),
  判官仍编造语义("绿色=正向反馈信号")并给出严格全序,且**从不主动说"分不出高下"**。
  故每个 judge-driven run 必须带一组"同轨迹不同渲染 / 退化输入"对照,测判官幻觉率;
  判官读图依赖 `claude -p` 的 Read 权限,图片路径必须在工作区内(`grpo_pixel.py` 的 `OUT=runs/grpo_pixel`
  是相对路径,换 cwd 启动会被拒 → 静默退化为 `fallback_milestone` 机器分,只数背包事件,不会让 run 失败——
  必须监控 `metrics.jsonl` 的 `fallback` 字段)。

---

## 7. 2026-07-10 增补:苦涩教训重设计已定稿,执行清单换新

当日与用户逐轮敲定的完整重设计在
**`knowledge/design_bitter_lesson_map_integration.md`**(§6–§11 为定稿部分),
其 §11 执行清单**取代**本单原有的模糊后续,但**不取代 §3 五个必修 bug(仍第一)**。
要点速览(细节勿凭记忆,读设计文档):

- 接口:弃 MiniLM/FiLM,subgoal 文本 token 直入 + hindsight relabel 造语言 BC 数据;
  aim 不做 ZOH,下发 tick 经 IPM 钉进北锚定地图;
- 视觉前端:~~DINO 替代 YOLOE 的探针门控提案~~ → **2026-07-10 后半用户直接拍板 DINO,
  YOLOE 整线废弃且已删码**(设计文档 §8 作废降级为 DINO 单臂探针 `tests/probe_dino_aim.py`);
- 时序:采样端帧堆叠 2–4(顺带消灭 T=1/T=64 失配);
- 慢塔会话:设计 2(无状态重提示 + 状态行/地图行外置),prompt 契约含
  prev_done/decision/subgoal/aim/done_when;Mamba 固定态判为吞吐性质非记忆
  (W4 三连败 + 混合架构自身即对冲),零样本无限流永不做;
- 慢塔选型:留 Omni(已验证资产 + 流式经济性);换塔触发条件与 Qwen-VL 备选见 §10;
- 训练:VPT BC 暖启动为主信号,GRPO 只做精修,判官落盘攒 RM 为后手。
