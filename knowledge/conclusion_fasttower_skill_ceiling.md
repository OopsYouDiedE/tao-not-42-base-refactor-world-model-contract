# 快塔学习上限地图:视觉 BC 沿技能课程的闭环成功率(2026-07-07)

> 终审=键鼠闭环技能成功率(非代理)。管线:ZEROCOPY GPU 渲染(:1)→ 教师(raycast 闭环)采 30 局
> 示范 → encode_feats(dinov3 CLS)→ train_fasthead BC 2500 步 → skill_ceiling 闭环 eval 16 局。
> 快塔=BCPolicy(冻结 dinov3 CLS + 因果 Transformer + 相机/键头),纯视觉闭环。

## 结果
| 技能 | 教师(环境天花板) | 快塔(视觉 BC) | 目标视觉 |
|---|---|---|---|
| chop_wood    | 0.967 | 0.500 | oak_log 纹理鲜明 |
| mine_stone   | 1.000 | **0.312** | 石头贴**石头后墙**=无对比 |
| mine_iron    | 0.800 | **0.688** | iron_ore 斑点鲜明 |
| mine_diamond | 0.667 | 0.500 | diamond_ore 鲜明,教师本身也只 0.667 |

## 结论
1. **"瞄准+挖"技能族,快塔从视觉 BC 全学得会**(0.31–0.69),**族内无硬天花板**——包括钻石。
2. **成功率跟"目标视觉可分辨度"走,不跟任务难度走**:iron_ore(鲜明)0.69 > 钻石/木头 0.50 >
   **石头贴石头墙(无对比)0.31**。感知(能不能看见目标)才是瓶颈,不是"挖什么"——印证 Step2/任务2。
3. **干净示范量关键**:同一 iron,新采 24 局重训 = 0.688,远超旧 ftt_c2bc(500 步)的 0.25。
4. 真正的天花板不在采矿族内,而在**技能模态**:aim+attack 可学;**合成(开背包 GUI)/盖屋(精确放块)
   是另一种动作模态**,V2 动作空间能否表达 + 快塔能否学,是下一个待测前沿(本轮未做)。

## 方法学要点(承教训)
- 全程闭环成功率终审;教师=raycast 特权,故本实验只证"快塔能从视觉学会 aim+attack 族",
  不证"慢塔 belief 有用"(那条见 [[conclusion_fovea_ceiling_mamba_seed]],去泄漏后近真空,受特权教师混淆)。
- chop_wood 曾因没给斧记为 0(RAW),给 wooden_axe 后教师 0.967——**设置 bug 会伪造 0 天花板**,已修。
- 渲染:ZEROCOPY(:1,需 root 起 `Xorg :1 -config xorg.conf.headless -ac`)obs['rgb']=CUDA 张量,
  boot 13s/17.9fps,快于 RAW(:99,39s)。

## GRPO 微调:能不能把快塔从 BC 往教师推?(train/fovea_twotower/grpo_skill.py)
BC 暖启动 → 随机 rollout → 组内相对优势 A=(R−mean)/std → 策略梯度(只训时序头),奖励=闭环挖到数。
| 技能 | BC(视觉) | GRPO best | 教师 | GRPO 效果 |
|---|---|---|---|---|
| chop_wood(oak_log 可见) | 0.50 | **0.81** | 0.97 | **↑ 往教师推**(rollout succ 0.5–0.75,奖励信号足) |
| mine_stone(石头贴石头墙,不可见) | 0.31 | **0.06** | 1.00 | **↓ 退化**(rollout succ 0–0.25,奖励稀疏,组内常全 0 无梯度) |
| mine_iron | ~0.9(greedy) | — | 0.80 | 近天花板,headroom 小,未细跑 |

**受控对照结论:GRPO 能否起效由"目标是否视觉可见(能否产生奖励信号)"闸门决定,不是 RL 本身行不行。**
可见目标(木头)GRPO 从 0.50 推到 0.81;不可见目标(石头)反而退化。→ **感知是硬性瓶颈,对策是改进感知
(给对比/YOLO 目标框),不是加 RL**。这是对"aim+attack 能学、瓶颈在感知"最强的一次闭环验证。

## 导航 / 合成:aim+attack 之外的动作模态(tests/integration/nav_skill.py)
- **导航**(走到可见 glowstone 地标 3 格内,带横向偏移须转向):**教师 0.90,快塔视觉 BC 0.611**。
  快塔学得会导航(看地标转向+前进)——可见亮地标,与"可见=可学"一致。运动模态(forward+camera_yaw)
  在 aim+attack 之外,但快塔一样能从视觉学。
- **合成**(planks/table/pickaxe):**能做,已验证**(`tests/integration/craft_skill.py`:5 oak_log →
  20 oak_planks,两次独立跑均 PASS)。破解的 GUI 操作机制(mod MouseInfo.kt / MinecraftEnv.kt:464-477):
  attack=左键、use=右键、**sneak=Shift**、`camera×20/3=6.67`→光标像素;**点击必须与 camera 移动同帧**
  (moveMouseBy 先于 onAction,光标才落槽);**sneak+attack 点输出=shift-click 自动合成全部直送背包**
  (绕开逐格存取的光标漂移);开背包光标=屏幕中心 (320,180);扫掠点覆盖 ±px 漂移保稳定复现。
  → **合成有可脚本化教师,示范工厂能产 demo**。("动作模态天花板"结论作废:GUI 合成同样可脚本化;
  快塔能否从视觉学 GUI 操作是下一问,但示范这一环已通。)

## 学生能否学会教师的动作?(终审:闭环成功率)
`tests/integration/{craft_ceiling,nav_skill,skill_ceiling}.py`:教师采 demo → BC → 学生闭环 eval。

| 技能 | 学生视觉BC 闭环 | 教师 | 判定 |
|---|---|---|---|
| 采矿 aim+attack(iron等) | 0.31–0.69 | 0.8–1.0 | **学得会**(反应式、有容差、目标可见、动作密集;GRPO 可推到 0.81) |
| 导航(走到 glowstone 地标) | 0.611 | 0.90 | **学得会**(反应式、地标可见) |
| **合成(GUI,oak_log→planks)** | **0.000**(BC holdout loss=0.0013) | 1.00 | **学不会** |

**合成学不会的精确诊断**(教师 40 局 100%,BC loss 0.0013 近乎背下来,闭环仍 0%):学生复现了
**常见运动背景**(开背包 1 次、小幅移光标 1.9°/步、attack)——帧多动作密,BCE/CE 平均学得好 → loss 低;
但合成成败系于**稀疏+精准的关键动作**:① **sneak(shift-click 那一下,整条 demo 仅几帧)** BCE 平均概率
<0.5 → 学生**从不触发**;② 精确**光标落槽**(零容差,BC 微误差沿 ~50 步累积→偏出槽);③ GUI 帧近乎
静态(DINO-CLS 几乎不变)→ 学生分辨不出当前步。→ **平均 BC 只学到背景,漏掉稀疏关键动作**。

**这同时命中两条教训**:#1 代理指标(BC loss/探针)≠闭环行为;Step2 关键变量是稀疏离散事件、平均重建
目标学不到。合成不是"示范做不了"(教师 100%),也不是"动作表示不了"(每步相机±1.4°在量程、sneak/inventory
键都在),而是**"稀疏+精准+零容差+帧不可分辨"这类动作,视觉平均 BC 学不会**——需序列/状态可分辨的表征
(如显式光标/槽位 token)或对稀疏关键动作加权/DAgger,是下一条路。

## 多步复合技能 + 同存档(tests/integration/multiskill_ceiling.py)
"挖铁→开背包合木板"多步任务(移动/瞄准/挖掘/GUI 多动作类型),**同存档**(固定 seed + 确定性命令;
craftground 禁存盘 SaveWorldMixin,复现靠 seed+命令重生成,非存/读档)。教师+学生用同一 SCENE 命令。

| | 挖矿 | 合成 | 两步都成 | 说明 |
|---|---|---|---|---|
| 教师 | 0.90 | 0.95 | 0.90 | 脚本化多步复合成立 |
| 学生 BC | **0.69** | **0.00** | **0.00** | BC holdout loss 0.0132 低,合成仍 0 |

**结论(整合全部发现)**:在**完美对齐的同一存档、同一多步任务**上,学生学会**反应式挖矿**(0.69)、**完全学不会
精准稀疏的 GUI 合成**(0.00)。→ **合成的失败是动作类型内在的**(episode 内累积误差 + 稀疏关键动作 sneak +
帧不可分辨),**与初始状态对齐无关**——同存档能保证同分布训练/评测,但无法改变这类动作的可学性(印证 [[fovea-cleanup...]]
教训:别把"同存档/低 loss"当行为有效)。同存档机制:craftground 用 seed+命令而非存档(禁存盘=吞吐+隔离+崩溃安全)。

## VPT 人类视频 BC 暖启动:PixelTower 离线可学性(2026-07-10,L4 单卡)

> 上文 GRPO 受控对照的"BC 暖启动"来自 raycast 教师示范(该感知路线已退役);本节
> 把暖启动换成 **VPT 人类视频**(learning 支柱,设计文档 §7 E1),供 grpo_pixel
> `--init-from` 精修。训练器 `train/craftground/bc_vpt_warmstart.py`;编码契约与采样端
> 由 `train/craftground/action_contract.py` 单一定义锚定(单测 5 项)。

- 数据:OpenAI VPT 承包商录像(公开 blob),滚动无限池下载(6xx–10xx 五索引、
  80GB 磁盘窗口、seen 去重):第一批(run1–3)收尾 58 clips,滚动池阶段(run4/run5)
  收尾 119 clips(下载器仍在攒);holdout 独立 2 clips(GUI 帧两侧一致剔除后 10733 ticks)。
- 编码:相机 mouse px × 0.15 deg/px(上游数据集格式常量;光流自标定被静态覆盖层
  污染的负结果见 lessons_do_not_retry)→ /18° → mu-law 11 bins;键位 VPT→V2 置换;
  T=1+帧堆叠 S=4;goal=零向量;prev 50% 置零。
- 吞吐:8.6–10.3k ticks/s @ GPU 90%+(帧堆叠必须 GPU 侧索引,CPU 展开会使 GPU 闲置,
  见 commit 10156e4)。

| 口径(holdout,GUI 剔除) | 多数类基线 | run3@600(58 clips) | run5@3000(119 clips,现 canonical) |
|---|---|---|---|
| cam top-1 acc(全 tick) | 0.8225(恒零 bin) | 0.8300 | **0.8316** |
| cam acc(非零相机 tick,占 0.1775) | 0 | 0.1617 | **0.1808** |
| 键 F1(有支持键均值) | 0(全不按) | 0.6508 | **0.6751** |
| holdout ce+bce | — | 0.6350 | **0.6148** |

run5 主要键位 F1:forward 0.938、sprint 0.972、jump 0.882;attack 升到 0.304 但仍低,
稀疏键归慢塔的结论不变。

主要键位(P/R/F1@正例率):forward 0.90/0.97/**0.935**@52%、sprint **0.966**@34%、
jump **0.904**@4.2%、back 0.71、right 0.68;**attack 0.08/0.86/0.15**@0.4%、
**inventory 0.00**@0.06%——稀疏键学不到,与上文"稀疏+精准+零容差动作视觉平均 BC
学不会"的结论一致(该职责归慢塔)。

五次 run 的训练动力学(数据流入带宽是主约束):run1 恒定 lr 3e-4 在 step 600 后 holdout 单调
恶化(0.66→0.93);run2 加 cosine 衰减到 0.6597@1200;run3 数据池扩到 58 clips 后
0.6350@600,15000 步末端过拟合到 2.77。滚动无限池阶段:run4(lr 2e-4)20k 步恶化到 1.7,
止损存档;run5(lr 5e-5 持续学习,池增到 119 clips)**0.6148@3000,全线最优**,其后
35 次 eval 无一改善,@38000 回升到 1.2–1.5,过拟合确立,run 终止。规律:扩池把最优点
从 600–1200 步推迟到 3000 步,方向有效;但训练消耗与下载流入之比约 200:1,超过最优点
后的训练只是在记忆当前池。**继续吃数据规模红利需要更大的起始池或更高下载带宽,
拉长步数无效。**

局限:① goal 全程零向量,FiLM 通道未训练,hindsight relabel(§12-5 后半)未做;
② 离线 acc ≠ 闭环有效(上文 D 曲线教训)——闭环必须过 CraftGround `--init-from`
GRPO 实测;③ holdout 仅 2 clips(6xx 风格),跨承包商泛化未测;④ 相机非零 acc
受人类微动在近零 bins 间混淆主导,是否够用由闭环判。
checkpoint:`runs/checkpoints/bc_vpt/best.pt`(canonical;2026-07-10 起=run5 best 的拷贝,
HF 归档 `unjustify/vpt-bc-pixeltower-v1-run5`;gitignored,产物不入库);
用法:`grpo_pixel.py --init-from runs/checkpoints/bc_vpt/best.pt`。

## 下一步(按终审优先级)
1. **GRPO 把快塔从 BC 的 0.3–0.7 往教师 0.8–1.0 推**——暖启动 checkpoint 已由 VPT BC
   供给(`--init-from`,本页上节);fovea 教师 BC 路线已退役。
2. 加**合成/盖屋**技能,探 aim+attack 之外的动作模态天花板(很可能才是真上限)。
3. stone 低分验证感知假设:换有对比的后墙重测,若跳升则坐实"视觉可分辨度=瓶颈"。
