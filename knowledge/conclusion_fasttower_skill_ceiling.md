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
可见目标(木头)GRPO 从 0.50 推到 0.81;不可见目标(石头)反而退化。→ **感知是硬墙,救法是改感知
(给对比/YOLO 目标框),不是加 RL**。这是对"aim+attack 能学、瓶颈在感知"最强的一次闭环验证。

## 导航 / 合成:aim+attack 之外的动作模态(tests/integration/nav_skill.py)
- **导航**(走到可见 glowstone 地标 3 格内,带横向偏移须转向):**教师 0.90,快塔视觉 BC 0.611**。
  快塔学得会导航(看地标转向+前进)——可见亮地标,与"可见=可学"一致。运动模态(forward+camera_yaw)
  在 aim+attack 之外,但快塔一样能从视觉学。
- **合成**(planks/table/pickaxe):V2 动作空间**有** inventory/use/drop/hotbar/camera,理论上能表达
  (开背包→相机当光标→attack/use 点击 2×2 格),但这是**盲操 GUI**(无光标位置反馈、依赖布局分辨率),
  **没有可脚本化的特权教师**(raycast/xyz 给不出 GUI 光标目标),连 VPT 都难。→ **示范工厂给不出 demo,
  当前"教师→BC→GRPO"管线够不到合成**。这是**动作模态的硬天花板**:aim+attack/运动可脚本化→可学;
  GUI 合成不可脚本化→本管线学不了(需 VPT 式人类演示或 GUI 光标级动作监督,是另一条路)。

## 下一步(按终审优先级)
1. **GRPO 把快塔从 BC 的 0.3–0.7 往教师 0.8–1.0 推**(尤其 stone:换非石头后墙给对比,或加 YOLO 目标框)。
2. 加**合成/盖屋**技能,探 aim+attack 之外的动作模态天花板(很可能才是真上限)。
3. stone 低分验证感知假设:换有对比的后墙重测,若跳升则坐实"视觉可分辨度=瓶颈"。
