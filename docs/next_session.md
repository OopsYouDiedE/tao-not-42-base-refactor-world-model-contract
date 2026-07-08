# 今晚定标作战单(2026-07-08 深夜写,给切换到 Opus 的训练会话)

> **使命:今晚这几轮 3090 训练,把未来大规模(集群)训练的全部设定钉死。**
> 背景与定案:记忆 `fovea-route-decisions-2026-07-08` + 
> `docs/architectures/fovea-hypothesis-verification-2026-07-08.md`(已验假设勿重验)。
> 每轮跑完:结果写回本文档对应栏 + 验证档案 + 提交 git。旧版本单(07-02 Colab)已废,git 可查。

## 已验完毕、直接当结论用(勿重跑)

心跳 85ms@1.5B / 94ms@2B-VL(448×256);I_gui=16×16 灰度+逻辑回归 99.91%;
IPM 数学精确;稀疏键(E 等)归慢塔;校准向量与文本 PE 基近正交(适配器须换基,
blocked on 类对数)。

## GPU 排程(3090 串行;R-E 走 CPU 渲染并行;R-F 晨间)

R-A 扩类校准(~1.5h) → R-B 分块 BC 三臂(~4h) → R-C 心跳 SFT(~1h)
→ R-D VL LoRA 冒烟(~0.5h) → R-G scaling 拟合(分钟级)。
闭环评测(eval_track_cmd / rollout smoke)用 Xvfb :99 CPU 渲染,可与 GPU 训练交错。

---

## R-A 扩类校准:log 进 bank(解 R2 wood_rate=0 + 定"每个新类"的标准配方)

**定的设定**:扩类标准流程(数据配比/neg_frac/epochs)——未来每个新类照此办理;
同时产出第 4 对 (文本PE, 校准向量) 喂 V10。

1. 向量:用 wood_gt(+wood_sm5p/sm6p 冒烟集)按 g1 配方拟合 log 类向量,并入
   bank(g1_vectors_v2.pt,WOOD_CLASSES 序)。注意 `wood.py` 头注:calib_nat_neg
   对 log 有毒(树未标注),**排除**;wood_negcert(认证无树负帧)代替。
2. conv 头 v7:`train/fovea_twotower/train_conv_head.py --data_dirs
   runs/data/calib640_rand{,2,3} runs/data/calib640_hardneg runs/data/wood_gt
   runs/data/wood_negcert --test_dir runs/data/wood_gt_hold --out
   runs/g1_conv_head_v7_wood.pt`。若脚本不支持第 4 类,标签工厂用
   `wood.py::wood_label_img`(8 角凸包投影)改造,GT 布局见 collect_wood_gt.py。
3. **闸门**:log 留出 mIoU ≥0.35 且 iron/coal/dirt 回退 ≤0.03(对照 v4 的 0.530)。
4. rollout 集成冒烟:grpo_rollout_worker 换 WOOD_CLASSES+v7 头,单 worker 8 局,
   **wood_rate>0.25** 即宣告 R2 感知瓶颈解除(下一轮 GRPO 有粮)。

**▶ 结果(07-08 深夜执行,commit 70c556a/8c7244b)**:
- 定标配方 = v4 铁/煤/土原班(calib640{,_rand,_rand2}+hardneg+motion_frames)+ wood_gt(log正)
  + wood_negcert(log负),neg_frac 0.35,**20 epochs**,ConvSegHead ncls=5;
  calib_nat/calib_nat_neg 排除(前者未入 v4 铁口径、后者含未标注树对 log 有毒)。
  工程修:load_eps 容忍缺 ray_xyz;wood_label_img 课程类前脸/log 凸包分派(保 iron/coal/dirt
  标签与 v4 逐像素一致)。产物 runs/g1_conv_head_v7b_wood.pt。
- 离线门(eval_wood_head):**G-W3 课程回归 PASS**(regress −0.014,iron 0.606/coal 0.588/
  dirt 0.605,dirt 反升);**G-W1 log mIoU 0.322**(<0.35 名义门,受 8 角凸包 GT 系统性
  高估的上限卡着,非训练不足——loss 已在 0.16 收敛)。
- **闭环 wood_chain(真 MC,v7b)决定性发现**:saw=27~174/500 → **感知已解**(学生看得见
  log token);但 **latch=0 三局全零 → wood_rate=0**。根因从"感知盲"移到"挖掘/导航执行":
  v17 平墙学生压不住 1 格宽橡树干+叶冠挡 raycast。**记忆 fovea-acceptance-loop 里的(b)
  "木目标感知盲"结论已被推翻**。集成已接(worker 换 v7b+WOOD_CLASSES+iron|log 闩锁),
  R2 现有木感知信号;wood 实际入包待 A1 树接近技能。
- 顺带根治:干净 env.close() 零 JVM 泄漏(实测),孤儿只来自异常/中途退出漏 close()→
  worker/wood_chain 加 atexit 兜底(SIGKILL/OOM 残留需驱动层 reaper)。

## R-B 分块 BC:k 定标(V7;大规模训练最重要的单一设定)

**定的设定**:chunk 大小 k、块内损失权重、22M 配方在分块下是否需调。

1. 改 `train_track_cmd.py`:动作头输出 k 步(相机 bins×k + 键×k),`--chunk_k` 旗标,
   块内损失默认均匀权重(不引入折扣,失败再议);数据侧现有示范逐窗重切标签即可
   (逐 tick 记录,天然支持任意 k)。**保持 v17 配方其余不动**(bin 逆频权重帽 3×/
   prev_dropout 0.5/switch_os 0.5,--d 512 --layers 7,数据=v17 同源 QC 集,
   查 runs/trackcmd_bc_v17 的 args 记录确认 data 目录)。
2. 三臂:k=1(重构后对照,防实现回归)、k=4、k=8。各
   `--total_steps 3000 --run_dir runs/trackcmd_bc_chunk{k}`。
3. 离线闸门:holdout 相机 CE/键 F1 与 v17 差 ≤5%(k=1 臂必须过,否则实现有 bug)。
4. 闭环闸门:`eval_track_cmd` student 臂,**到达≥0.47 且 切换≥0.46×教师**
   (v17 口径);挖掘持续段完整率(attack 连续≥23tick 的段占比)一并记录。
5. **决策规则**:取通过闭环闸门的最大 k(并列取 k=4);全不过→k=1+复盘,
   大规模训练回退逐 tick。**k 一旦定下,写死进集群任务书。**

**▶ 结果(07-08 深夜,commit ee55896/103084b)——裁决 k=1,分块对反应式快塔否决**:
- chunk_k 落地(头出 k 步/块内均匀权重/首步评测),k=1 与旧口径逐字兼容(v17 ckpt 加载
  推理验过,保 R-A rollout);StudentPolicy 分块开环执行(边界解 k 步入缓冲后回放)。
- 3 臂(v17 配方 d512/L7,3000步;full 14000步复跑证 3000 已收敛,cam_acc 0.275=0.275):
  离线首步 cam_acc k1 0.275(≈v17 0.273)/k4 0.270/k8 0.256(−6.9% 破 5% 门,出局);
  闭环(真MC同seed课程,tracking):arrive k1 0.625/k4 0.125/k8 0.714*(n=8噪),
  **switch k1 0.125 > k4 0.125 > k8 0.000**(开环回放对局中换目标不可反应)。
- **裁决:chunk_k=1**。两条硬信号(离线首步准确率随k单调降 + 闭环 switch 随k崩)一致;
  分块想要的"挖掘持续段"属重复子动作,该由挖掘闩锁/宏承担,不该分块整个反应式策略。
  代码保留(k>1 可用)。集群任务书写 **k=1,块内均匀权重**。

## R-C 心跳微决策 SFT:慢塔 A1 行为(冻结数据格式=未来一切慢塔的输入模板)

**定的设定**:状态行 schema(VL/Omni 同款沿用)、微决策词表、僵局阈值 N、
1.5B LoRA 配置。

1. 新建 `train/fovea_twotower/heartbeat_sft.py`,合成轨迹混合真实 R2 rollout 记录
   (rec 里 inv_steps/goal_log/vis 序列)。**冻结格式**:
   状态行 `t=<tick> 库存:<item×n,…|空> 可见:<cls(dist格)|无> 位移:<m> goal:<cls>`;
   决策词表 `{继续, 换目标:<cls>, 重规划}`。
2. GT 规则:新里程碑→重规划;连续 N=20 次心跳库存无Δ且位移<阈→重规划(僵局);
   其余→继续。N 作 sweep {10,20,40} 各训一份小样,取留出准确率最高者定 N。
3. LoRA 照 reason_delta 配方(r16 qkvo),底座 Qwen2.5-1.5B + 已有 v4 adapter 续训
   或并行新 adapter(推荐新 adapter,防复核能力回退;跑完重测 reason_delta 留出)。
4. **闸门**:留出决策准确率 ≥0.95、格式合规 1.0、reason_delta 复核评测不回退。

## R-D VL-2B LoRA 冒烟:A2/集群 VL SFT 配置模板

**定的设定**:Qwen2-VL LoRA targets/r/lr、grad-ckpt、batch、图像分辨率(448×256)。

1. 玩具集 200 样本:craft/track 示范帧(448×256 重采样)+R-C 同款状态行→决策文本。
2. peft LoRA r16 targets `q_proj,k_proj,v_proj,o_proj`(视觉塔冻结),bf16,
   grad-checkpointing on,batch 1×梯度累积 8,lr 1e-4,200 步。
3. **闸门**:loss 降 ≥50%、显存峰值 ≤20GB、无 NaN。数字记入配置模板表。

## R-E GUI 高清采集 + V8(CPU 渲染,与 GPU 训练并行)

**定的设定**:GUI GT 工厂配方;热键栏在 GUI 内的掩码归属(数据仲裁)。

1. 扩 collect 脚本:640×360,fast_reset 后 give 已知物品→按 E 开背包→录帧+
   `gui` 标志+槽位坐标表(容器布局硬编码常量=GT,零人工)。≥20 局、含空背包负例。
2. V8:YOLOE 文本 PE 对物品图标零样本打分("iron ingot"/"oak planks"…),
   记录逐类 AP;<0.3 则结论"GUI 页须走域内校准"(R-A 配方复用)。
3. 高清版 I_gui 复验(16×16 逻辑回归重训一次,预期 ≥99.9%)。
4. 顺带录一段人工/脚本在 GUI 内按数字键换热键栏的交互,仲裁掩码表。

## R-F Nano2-9B QLoRA 冒烟(下载已在后台,晨间跑)

**定的设定**:混合架构本地工具链风险清零(集群租卡前置条件)。
NF4 载入(~5GB)+LoRA(in/out_proj+qkvo)50 步 dummy 文本反向。
**闸门**:backward 通过、显存<24G、mamba-ssm 2.2.4 内核无报错。

## R-G scaling 拟合:集群预算数字

R-B 落地后,`train/fovea_twotower/scaling_fit.py` 吃 c2_size_{s,m,l,l2,xl} +
c2_demo_{24,48,96} + 今晚分块臂 → 外推"到达 0.8"所需示范局数×参数量,
**产出集群 C2 任务的两个预算数字**(目标示范量、目标模型规模)。

---

## 大规模训练设定总表(跑完填,这就是交付物)

| 设定 | 由哪轮定 | 值(待填) |
|---|---|---|
| 快塔 chunk k / 块内权重 | R-B | ✔ **k=1**(分块否决:离线首步acc随k降+闭环switch随k崩);块内均匀权重 |
| 快塔规模×示范量预算 | R-G | ✔ 甜点 ≤22M(外推~28M)@83局;容量杠杆封顶~2°(距教师天花板);示范曲线太噪(arrive 0.18→0.58→0.42非单调)定不出局数→**主杠杆=移动天花板(GRPO/更好教师/数据),非买容量** |
| BC 配方(权重帽/dropout/OS/DAgger β) | R-B(默认沿 v17,回归才动) | 沿 v17:帽3×/prev_drop0.5/switch_os0.5/d512·L7 |
| 扩类校准配方(数据配比/neg_frac/闸门) | R-A | ✔ v4铁班+wood_gt+wood_negcert;neg_frac0.35;20ep;ncls=5(课程回归PASS/log0.322上限受凸包GT限) |
| 状态行 schema / 微决策词表 / 僵局 N | R-C | |
| 慢塔 LoRA(r/targets/lr) | R-C(1.5B)+R-D(VL) | |
| VL 图像分辨率/批量/grad-ckpt | R-D | |
| GUI GT 工厂配方 / 掩码表 | R-E | |
| 混合架构工具链(本地侧) | R-F | |
| 判官设定(4条/组排名制,全量落盘) | 已定(R2),不动 | ✔ |

## 纪律提醒(与用户当面定过的,违者重来)

- 教师必须是学生观测的函数;教师必须确定性;特权信息只进训练侧。
- 每级升级要有上一级在给定预算内的证伪记录,不许跳级。
- 判官调用全量落盘(蒸馏本地判官的数据在攒)。
- 运行时零脚本:宏只活在采集器;闩锁不进部署回路(R-A 集成冒烟里 rollout worker
  的 raycast 闩锁暂留=已声明脚手架,A1 收口时移除)。
