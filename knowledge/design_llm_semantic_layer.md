> ⚠️ 局部废弃(2026-07-10 封存):§1-§2 裁决(连续标定不归 LLM / 离散绑定归 LLM)仍有效,且被 conclusion_omni_pixel_control 实证背书;§3 起的执行序(PPO+AD MVP / task_text / gaming500 hindsight)属退役世界模型线,勿当待办。

# 设计结论:LLM 语义层接入(2026-07-03 第二场辩论收敛)

> 议题:借用 LLM 因果推理指导 ActionHead。经正反两轮辩论收敛。
> 相关:mental_world.md §6(北极星)、design_wca_agent.md(A 只吃 W 隐状态硬约束)、
> conclusion_g500_gates_probe.md(今日闸门实测)、train/minecraft/task_text.py(条件通路,已随 prune3 删除)。

## 1. 裁决

- **否决**原始形态:LLM 输出/指挥 30Hz 动作序列、LLM 直接指挥学习式 ActionHead。
  双方一致:LLM 因果推理只在符号/语义层被验证;Voyager 的低层是确定性脚本 API 而非
  学生网络;推理期任何形态的 LLM 不得进 66ms 控制环。
- **采纳**收敛形态:LLM = **边缘文本编译器**——训练期离线 hindsight 重标注(VLM 对
  10-30s 片段做事后描述→冻结 MiniLM 384 维→ActionHead 条件 token),推理期至多
  episode 边界一次"归因+计划重编译"(可断网降级为静态计划)。模型内只见冻结向量。
  先例:STEVE-1(文本嵌入条件化 VPT 键鼠策略)、GameGen-X InstructNet、Dynalang。

## 2. 因果分工线(硬边界)

| 归 LLM/文本 | 归 latent WM + 反应头 |
|---|---|
| 离散绑定(键→功能,说明书是原生存储格式) | 连续标定(灵敏度/相机平滑,只存在于跨帧 patch 网格,Gate 0 D3) |
| 科技树/任务依赖/episode 间试错归因 | 瞄准/走位/跳跃时机(禁入) |
| 新游戏 DAG 生成器(消掉逐游戏人工成本) | 重映射 dose-response(北极星主线) |

## 3. 本轮核实的仓库硬事实

1. `train/craftground/env.py::DISCRETE_TO_V2`:27 动作无 inventory/craft/GUI——
   **agent 当前物理上不能合成**;smelt_iron 之后的成就瓶颈是**执行器缺失**而非探索
   指数问题(修正 conclusion_craftground_run 的归因);`spaces.py` 的 ACTION_NAMES
   注释表与实际映射不符,应修。
2. CraftGround 首轮 4/16 基线**不可用作对照臂**:权重丢失、无随机基线(其结论
   文档自评"没有它所有成就数无意义")。
3. `train_ppo_ad.py` 无文本条件入口;task_text.py 在 dreamer4/BASALT 栈——LLM
   条件 MVP 的管道尚未接线。
4. hindsight 重标注 500h 全量 ≈ 6-18 万片段/数百美元级,须 10h 试点先行。

## 4. 执行序(合并案,按依赖)

1. **现在(L4,零新依赖)**:全 121-token 空间 CAD/方向绑定复测(latent 文本化的
   闸门钥匙)+ task_text 条件通道活性压测(通道死则整条语言线冻结)。
2. **并行小额**:10h hindsight 标注试点——1% 人工抽检 + 标题-动作互信息闸门,
   通过才放量(VLM 系统性幻觉是污染不是钝化,不能靠"噪声退化为无条件 BC"兜底)。
3. **CraftGround MVP(约 4-6 周,非两周)**:补随机基线→重跑裸 PPO(存 checkpoint)
   →动作空间扩展+宏执行器作为独立工程项预验收→对照臂:A 裸 PPO / B+ 脚本课程+执行器
   (增强版 B,LLM 必须超过它而非裸 PPO)/ C 虚构重命名版防先验泄漏 / 处理组 LLM
   一次性编译计划。LLM 臂最后进场。
4. 可证伪判据:LLM 零人工生成的依赖计划 ≥ 手写 DAG 90% 成就进度;处理组>增强 B;
   写明两条失败判据:处理组≯B→降格课程生成器;通道死亡→实验无信息量,止损。

## 5. 红线(违反即回到否决)

- 推理期 LLM 进 66ms 环或成为 15Hz 头运行时依赖;
- gaming500 预训练阶段(只训 WM+CAD)为 LLM 线挪用 L4 算力;
- 以"外置文本记忆"替代/推迟 §6 的 h 记忆与重绑定主线(说明书是第二通道非替身);
- 北极星防火墙:重映射 dose-response 评估强制 LLM/说明书通道置零(空条件向量),
  语义轨单独评且含"错误/随机说明书"对照;LLM 线结果不得作为 §6 进展证据。
