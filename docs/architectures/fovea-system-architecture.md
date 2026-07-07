---
name: fovea-system-architecture
description: 凹视双脑系统结构参考:感知底座/快头/慢脑/宏层的代码路径、最优checkpoint、关键超参、证据指针——论文写作与后续训练的单一入口
metadata:
  type: project
---

# 凹视双脑系统:结构参考(2026-07-08 定稿)

**配套文档**:[[fovea-brain-division-scale-plan]](分工定案+数据引擎+集群就绪判据)、
[[fovea-experiments-index]](全部实验的判据/结果/产物索引,**含 07-08 结论强度审计
——本文引用的数字以该索引的强度标注为准**)、
knowledge/design_fovea_yolo_fasttower.md(感知设计与 G1 战役记录)。
**分工原则(用户 2026-07-07 定)**:慢脑=判断局势/规划收集/核查差额/记路线;快脑=其余一切。
**证伪史约束**:每个组件只承担有正面证据的职能——预训练 LLM 的 SSM 状态不再兼职视觉
世界记忆(边界:W4/W4b/W4c 在 LoRA/解冻递归张量+3000 步预算内未通,非永久不可改造);
情景记忆不学习、走显式注册表(S5 双塔复证)。

## 0. 总览(运行时回路,三时间尺度)

```
慢脑 LLM(低频/事件触发, Qwen2.5-1.5B + LoRA)
  读: 文本消息总线(库存/事件/注册表摘要/任务书+配方卡)
  写: ①感知点名(新类文本→PE→向量bank) ②策略指令(goal 类别) ③库存复核("已齐备")
      goal 在两次输出间零阶保持,异步不阻塞
        │ goal
        ▼
快头(每帧, TrackNav 2.4–22M)                统一感知头(冻结 YOLOE-26 + 校准头)
  读: goal 相对 token [K,8] + a_{t-1}   ◄──  640×360 帧 → pad384 → P3 嵌入
  写: 相机 mu-law bins ×2 + 20 键            → ConvSegHead 稠密分割 → 连通域
        │                                     → token [K, 6几何+C+1概率]
        ▼                                     (右下角手持物区遮罩)
宏技能层(脚本占位, 待学习技能的位置标记): 挖掘宏(raycast 闩锁)、GUI 合成宏(未建)
        │
craftground 环境(锁步: 1 step = 1 tick; 640×360 RAW 软渲染, Xvfb :99)
```

延迟总账(3090 实测):统一头 23ms + 22M 快头 5ms = **28ms/帧 → 36Hz**(MC 20Hz tick
的 2.4× 余量);慢脑秒级生成经零阶保持异步化。设备速度鲁棒性三探针见实验索引。

## 1. 感知底座:统一 YOLOE 头 + 域内校准

**代码**:`net/fovea_twotower/yolo_unified.py`(UnifiedYoloe26/pad384)、
`net/fovea_twotower/seg_head.py`(ConvSegHead + 投影 GT 工厂)、
`net/fovea_twotower/token_stream.py`(TokenHead:分割→连通域→token)。
**对拍**:`tests/probe_yoloe_unified.py`(打分重建 max|Δ|=0,V1/V2/V3 全 PASS)。

- 骨干 = YOLOE-26 promptable(`runs/checkpoints/yoloe-26l-seg.pt`),**全程冻结**;
  pf 版(`yoloe-26l-seg-pf.pt`)骨干非同权(词表融合后轻微调副本),仅开放集提案用。
  陷阱:`predict()` 会把文本向量融进卷积毁掉嵌入通道——嵌入必须直调底层 nn 前向。
- 打分数学:`score_i = BN_i(cv3_i特征)·w × exp(logit_scale_i) + bias_i`;
  `set_classes` 向量原样进 cv4(reprta 不参与)→ 文本 PE 与校准向量同空间互换。
- 校准三级(同一 bank):①域内 GT 学习向量(逻辑回归级,`runs/g1_vectors.pt`,
  [4,512]+bias);②文本 PE(新类冷启动);③conv 分割头(精确掩膜,
  **`runs/g1_conv_head_v4.pt`**,0.7M,配方=`train/fovea_twotower/train_conv_head.py`:
  epochs 10/neg_frac 0.35/数据含纯负房+闭环运动帧)。
- GT 工厂(零人工标注):setblock 已知坐标 + raycast 锚定 + 位姿针孔投影前脸
  (FOV 70°/眼高 1.62,raycast 核对 178/178);采集器
  `tests/integration/collect_calib640.py`(--rand_layout 反位置先验/--hard_neg 纯负房)。
- 核心类 `CLASSES = [iron_ore, coal_ore, dirt]`(单一定义:token_stream.py)。
- G1 终审:conv 头 mIoU 0.530/Δ+0.528 双 PASS(随机布局干净集,基线 0.001)。

## 2. 快头:goal 相对 token → 键鼠动作

**代码**:`net/fovea_twotower/yolo_parse.py`(TrackNavTower 结构)、
`net/fovea_twotower/token_stream.py`(goal_relative 折叠/TokenTeacher/AimTeacher)、
`train/fovea_twotower/train_track_cmd.py`(BC 配方)、`eval_track_cmd.py`(闭环四臂)。

- 输入契约:`goal_relative` 把 token [K,6+C+1] 折成 **[K,8]=[几何6, p_goal, p_other_max]**
  ——快头结构上类无关,"听指挥"内建在输入契约;切换指令=p_goal 列换列。
- 结构:TrackNavTower(goal 交叉注意选目标 + 因果 Transformer 干),d/layers 可扫:
  2.4M(256/3)/8.8M(384/5)/**22M(512/7,当前甜点)**/64M(768/9,125 局数据下回落)。
- 动作:相机 mu-law 11 bins ×2(CE,**CAM_NORM_PX=120**——教师 ±18°/步口径,
  勿用人类小步的 CAMERA_SCALE=10)+ 20 键 BCE(forward 正例权重 ×4)。
- BC 配方(v14 定型):bin 逆频权重加帽 3× + prev_action dropout 0.5 +
  切换窗口过采样 0.5 + DAgger 一轮(β=0.15)。
- **最优 ckpt**:`runs/trackcmd_bc_v17/best.pt`(22M×QC 数据:追踪 12.4° 超教师/
  到达 0.47/切换 0.46×教师);2M 参考 `runs/trackcmd_bc_v14/best.pt`。
- 示范工厂:`tests/integration/collect_track_cmd.py`(TokenTeacher 观测一致教师,
  开局扰乱+每 40±10 步切换,--store_frames 顺产感知训练帧,--dagger_ckpt DAgger 环);
  质检 `tests/integration/qc_demos.py`(教师段末>25° 自动拒收,实测拒 ~23%)。
- 五条铁律(证据见实验索引):教师须是学生观测的函数;教师须确定性(随机方向=
  不可观测潜变量,BC 均值归零);感知与策略都在闭环访问分布上迭代;纯离线 BC
  到不了闭环(DAgger 标配);校准 token 抬 BC(+10.6pt vs 未校准 +6.7pt)。

## 3. 慢脑:差额规划 + 库存复核

**代码**:`train/fovea_twotower/reason_delta_sft.py`(规则引擎生成+多解拓扑校验器);
前身 `reason_sft.py`(think 模式诱导,留出 think 率 0→1.00)。

- 底座 Qwen2.5-1.5B-Instruct + LoRA r16(qkvo,~9M 可训);**0.5B 不够、1.5B 够用、
  4B 本机上限**(W4 先例:3090 冻结+LoRA 可训)。
- 任务形态:给定库存 → 缺什么 → 按依赖序补齐计划(`<think>` 反向展开);
  42 物品科技树组合生成,样本无限。
- **定型结论:程序用 SFT 注入(分布内 1.00),事实(配方)走上下文配方卡
  (未见目标 0.73),版本热更零重训。**
- **最优 adapter**:`runs/reason_delta_lora_v4`(含"已完成"态:15% 样本目标在库存——
  C1b 教训,v3 拿着生铁仍说缺生铁);纯规划参考 `runs/reason_delta_lora_v3`。
- 运行时接口:`tests/integration/fullloop_chain.py::SlowBrain.plan(goal, inv)`
  → (计划步列表, 已齐备?, 原文)。

## 4. 宏技能层(脚本占位,诚实边界)

待学习技能的位置标记,地位等同 setblock 课程脚本;宏内部允许 raycast(学习策略禁止):
- 挖掘宏(`fullloop_chain.py` ②):raycast 命中铁矿 ≤5.5 格接管 → 贴近到 3.2(REACH)
  → 锁相机持续攻击 + 吸拾轻触。教训:接管范围必须覆盖教师停靠点(0.6 格真空=死锁)。
- GUI 合成宏:未建(GUI 槽位坐标确定性,可脚本化;旧 craft 实验闭环 0% 的已知深坑)。
- 去特权化阶梯:raycast 宏 → 纯 token 宏 → 学习出的技能。

## 5. 全回路装配(C1b,PASS 0.70)

`tests/integration/fullloop_chain.py`:慢脑计划(10/10)→ 执行挖铁(0.70)→
慢脑复核(条件 7/7)→ 视觉标记回家(条件 7/7)。家标记=煤矿石柱 2×2
(**材质须场景天然不存在**——dirt 被挖穿墙洞的天然泥土假冒过)。
执行器可选 teacher(装配闸门口径)/student(参考;学生冻结不动点残余归 GRPO)。

## 6. 谱系与历史组件(论文引用原样,不再改动)

- 双塔 Step1–5:`net/fovea_twotower/tower.py`(ContextTower/ActionTower,GDN 状态播种)
  + `train/fovea_twotower/train_w*/eval_s*`——播种传历史 PASS(keys +27% F1)、
  情景记忆 FAIL(S5);B0.5 对照仍欠。**慢脑换型(弃"预训练 SSM 状态当视觉记忆")的
  依据是双支柱**:①该路在测过的预算内未通(W4/W4b/W4c:LoRA、乃至解冻递归张量
  A_log/dt_bias/D,3000 步,age R²≈0——边界标注:非"永久不可改造",全参/更长预算
  未测);②纯文本职能已验证够用(E2:程序 SFT+配方卡上下文),不需要它通。
- 快头 BC 谱系:`net/bc/policy.py`(BCPolicy/CondPolicy/TextCondPolicy,DINO-CLS)、
  `net/fovea_twotower/yolo_parse.py` 的 TrackNav 初版(0.778>0.711)。
- 慢塔 LLM 探针:`train/fovea_twotower/reason_sft.py`(813125d)。

## 7. 参数/资产总账

| 组件 | 可训参数 | 冻结资本 | 最优产物 |
|---|---|---|---|
| 统一感知头 | 向量 2K + conv 头 0.7M | YOLOE-26 骨干 | g1_vectors.pt / g1_conv_head_v4.pt |
| 快头 | 2.4M–22M | — | trackcmd_bc_v17(22M)/v14(2M) |
| 慢脑 | LoRA ~9M | Qwen2.5-1.5B | reason_delta_lora_v4 |
| 合计可训 | **<32M** | ~1.6B | — |

集群第一任务(C2 裁决):快头×示范量联合放大;数据侧上限见 C3(N=2 并行 1029 局/时,
N=4 需压 JVM 堆 2G)。
