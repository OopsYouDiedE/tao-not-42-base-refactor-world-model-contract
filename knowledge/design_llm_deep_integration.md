# 设计:LLM 深度嵌合快反模型(异步双系统)(2026-07-03)

> 在 [design_llm_semantic_layer.md](design_llm_semantic_layer.md) 裁决基础上的方向升级(用户拍板):
> VLM/LLM 从"边缘文本编译器 + 判官"升格为**异步上位指导模型**,与快反模型(Actor-Critic-Dreamer)
> 深度嵌合。范式 = 双系统(慢系统 VLM 规划/评估,快系统 30Hz 反应),先例:Figure Helix、
> NVIDIA GR00T N1、Hi Robot(π0)、DeepMind SIMA。
> 原裁决**作废**条款:"episode 边界至多一次归因/重编译"。**保留**红线见 §5。

## 1. 物理约束(不随设计改变)

同步进 66ms 控制环不可行(VLM 推理百毫秒-秒级)。一切嵌合形态必须**异步**:
慢系统按自身节拍(0.5–2s)写共享接口,快系统每 tick 非阻塞读最新值;
延迟表现为**陈旧度**(staleness)而非阻塞。

## 2. 嵌合点(下行:LLM → 模型)

| 插槽 | 位置 | 状态 |
|---|---|---|
| S1 目标条件 actor | `net/dreamerv3/behavior.py::GoalActorHead`(文本点乘打分) | 已有 |
| S2 目标条件 critic | `ImagBehavior.value`(use_goal 时拼接 goal 嵌入) | 本轮接入 |
| S3 语义奖励头 | `net/guidance/heads.py::SemanticRewardHead`,经 `ImagBehavior.loss(reward_fn=…)` 混入想象回报 | 本轮接入(结构) |
| S4 规划器目标对齐 | `net/dreamerv3/planner.py`(α·余弦对齐项;value 同步目标条件化) | 已有/本轮同步 |

**硬边界:世界模型动力学不接受目标条件化。** 物理与目标无关;若把 goal 喂进
RSSM/SpaceTimeTransformer,WM 会学出"目标依赖的物理"幻觉,且破坏"换游戏只重学物理"
的快速适应目标。LLM 深入到行为层全部头,止步于动力学。

角色分工(与 2026-07-03 会话收敛一致):
- **LLM 管"分几步走"**(episode 级):子目标 DAG/计划编译,异步重编译;
- **VLM 管"每步走没走到"**(片段级):目标条件化成对偏好 → Bradley-Terry 蒸馏进 S3,
  里程碑完成判定驱动总线 advance;
- **Critic 管"步内怎么走划算"**(秒级):定义不变(λ-return 期望),接上 S3 后自动成为
  "判官分数的期望";长视野被子目标链切成 critic 射程内的短段;
- **物理合理性归世界模型自身损失**(KL+重建),不归 Critic、不归 LLM。

## 3. 运行时(异步总线)

`utils/guidance_bus.py::GuidanceBus`——线程安全最新值寄存器:
- LLM worker 线程 `publish_plan(subgoals)`(重编译)/ 里程碑判定后 `advance()`;
- 采样环每 tick `read()` 得 `Guidance(subgoal, goal_vec, source)`,goal_vec 直接喂 S1/S2/S4;
- **断网/陈旧降级**:LLM 计划超 `stale_after_s` 未刷新 ⇒ 自动切静态计划(source="static");
- 文本编码器(冻结 MiniLM,`train/minecraft/task_text.py`)依赖注入,总线不 import 模型代码。

VLM worker 本体(API 调用、抽帧条构造、偏好采集)尚未落地,契约:输入 = U1 抽帧条 +
U3 硬事实(成就/物品栏事件),输出 = publish_plan / advance / 偏好标签流。

## 4. 上行通道(模型 → LLM)与整合梯子

- U1 真实抽帧条(4–8 关键帧/片段);
- U2 **想象 rollout 解码帧**(dreamer4 `generate()` + tokenizer 解码):VLM 在"梦里"预演
  计划再下发——指导的对象是想象的走向,不是动作。前置依赖:流匹配采样(否则梦是糊的,
  VLM 无法评判);
- U3 硬事实流:凡环境有硬事实处用硬事实覆盖 VLM 判断,VLM 只填空隙。

执行梯子(逐档拿证据再上):L1 异步中途重规划(本轮全部结构就绪)→ L2 潜向量接口
(VLM latent 替代 MiniLM,加 adapter)→ L3 双向闭环(U2 梦境预演)。
L4(VLA 做策略本体)维持否决(30Hz + L4 算力)。

## 5. 红线与防线(承接原裁决)

- 推理期 LLM/VLM 不得同步进 66ms 环;快反头必须可断网降级(静态计划);
- gaming500 预训练期(只训 WM+CAD)不为本线挪 L4 算力;
- **北极星防火墙**:重映射 dose-response 评估强制语义通道置零——S3 的 `reward(feat, goal=None)`
  构造性返回 0,S1/S2 置空条件向量;本线结果不作 mental_world §6 进展证据;
- **Goodhart 防线**:VLM 偏好与人工偏好一致率常态抽检(试点闸门不达标即止损);
  优化目标是"判官分数"时,判官漏洞即攻击面——硬事实校准 + 虚构重命名对照(防先验泄漏)。

## 6. 训练管道(蒸馏,推理期零 LLM 依赖)

hindsight 片段标注(VLM 事后描述,复用原裁决管道)→ (片段, 子目标文本) 对:
1. S1/S2 目标条件化训练:想象 actor-critic 照常,goal 来自标注(见 train/crafter goal 通路先例);
2. S3 偏好蒸馏:同目标片段对 → VLM 成对偏好(交换顺序消位置偏置)→ BT 损失回归段内
   Σr̂ 差;训练循环在 train/(待接),头结构本轮就绪;
3. 全程闸门:10h 试点 + 1% 人工抽检 + 标题-动作互信息(原裁决条款不变)。

## 7. 本轮落地清单

- `net/guidance/`(config/heads):GuidanceConfig、SemanticRewardHead + 工厂;
- `net/dreamerv3/behavior.py`:critic 目标条件化(S2),`loss(reward_fn=…)` shaping 插槽(S3);
- `net/dreamerv3/planner.py`:value 打分同步目标条件化;
- `utils/guidance_bus.py`:异步指导总线(L1 运行时骨架);
- `tests/integration/test_guidance_build.py`:CPU 冒烟(形状/梯度隔离/防火墙置零/总线降级)。

待办(按依赖序):task_text 条件通道活性压测(第一闸门,通道死则全线冻结)→ VLM worker +
偏好蒸馏训练循环(train/)→ L1 端到端对照("异步指导 > 静态计划")→ L2/L3。
