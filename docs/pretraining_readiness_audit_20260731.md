# 正式训练前验收审计

审计日期：2026-07-31。本报告区分真实 Minecraft 执行、CPU 合成训练验证和正式 GPU 才能完成的验证。未通过的范围保持关闭。

## 结论

| 范围 | 状态 | 证据或限制 |
|---|---|---|
| Python 全量测试 | 通过 | `117 passed` |
| Python 语法编译 | 通过 | `compileall` 覆盖 `dataset/game_environment/tao/tools/train/tests` |
| 差异空白检查 | 通过 | `git diff --check` 无空白错误；仅有 Git 行尾提示 |
| Ruff | 环境阻塞 | 本机未安装，代理不可用导致无法下载；不能记为通过 |
| 玩家与静态世界快照 | 真实通过 | 连续 8 次恢复，位置、视角、背包、生命、饥饿、经验、状态效果一致 |
| 普通实体、计划刻、跨维度快照 | 未通过 | 当前实现不保存普通实体和计划刻，也不支持跨维度公平复位 |
| 观察驱动 `2+6` 执行 | 真实通过 | 2 条 Terra reference、6 条 Terra policy substitute，8 条均真实执行，起点偏差为 0 |
| `2+6` 训练数据加载 | CPU 通过 | 从真实 `execution.json` 和真实轨迹图片加载，严格验证来源数和图片存在性 |
| 联合目标反向传播 | CPU 通过 | 6 条 policy 进入直接优势加权项，2 条 reference 只进入行为克隆项 |
| PPO/GRPO 概率比 | 门禁生效 | 当前轨迹没有 `old_logprob`，加载器明确拒绝宣称 PPO/GRPO |
| 长任务续跑决策 | 单元验证通过 | `PROGRESSING` 扩预算并保存稳定检查点；无物理证据时不判 `INFEASIBLE` |
| 长任务 Minecraft 端到端 | 未通过 | 尚未真实跑通钻石、要塞等十几分钟任务的检查点续跑 |
| 多尺寸与 Gemma 4 预算 | CPU 合同通过 | 保持宽高比、48 像素对齐、raw patch 和 soft token 上限校验 |
| PRO 6000 利用率与并行训练 | 待正式机器验证 | 本机无真实模型，不能验证显存、吞吐、推理与训练流重叠 |

当前可以开始静态、同维度、无普通实体依赖课程的受限训练试验。不能开放实体战斗、船、村民、刷铁机、末影龙、红石/流体严格时序和跨维度公平 batch。

## 真实执行证据

起点观察。该图片直接来自真实 CraftGround 运行，不是 JSON 或示意图。

![真实起点观察](../runs/craftground-observation-curriculum-terra-2plus6-20260731-retry10/initial.png)

参考轨迹 R01 起点。

![R01 起点](../runs/craftground-observation-curriculum-terra-2plus6-20260731-retry10/R01/start.png)

参考轨迹 R01 执行到第 64 tick。

![R01 第 64 tick](../runs/craftground-observation-curriculum-terra-2plus6-20260731-retry10/R01/tick_064.png)

参考轨迹 R01 终态。其目标距离从 `14.543` 降到 `7.386`，净进展 `7.157`。它没有达到本轮成功阈值，因此状态是 `PROGRESSING`，不是成功，也不是不可行。

![R01 终态](../runs/craftground-observation-curriculum-terra-2plus6-20260731-retry10/R01/tick_124.png)

策略替代轨迹 P01 的真实终态。

![P01 终态](../runs/craftground-observation-curriculum-terra-2plus6-20260731-retry10/P01/tick_096.png)

完整 52 张图片、八条轨迹和逐条相对优势见 [观察驱动 2+6 执行报告](../runs/craftground-observation-curriculum-terra-2plus6-20260731-retry10/REPORT.md)。Markdown 图片链接检查结果为 `52/52` 存在。

## 训练目标边界

| 数据来源 | 数量 | 直接优势加权 | 行为克隆 | PPO/GRPO 概率比 |
|---|---:|---:|---:|---:|
| `reference_expert` | 2 | 禁止 | 允许 | 禁止 |
| `policy_sample` | 6 | 允许 | 默认禁止 | 仅保存生成时 `old_logprob` 后允许 |

本机 dry-run 使用可微序列负对数似然验证梯度边界。它证明掩码和反向传播合同正确，不证明 Gemma 4 已完成正式微调。正式训练入口仍需把模型 token loss 接到该合同，并记录生成策略版本、采样参数、token 掩码和 `old_logprob`。

## 图像与动作合同

Gemma 4 使用 `16×16` patch，视觉池化核为 `3×3`。`640×360` 在示例缩放下对齐到 `1056×576`，得到 `66×36=2376` raw patches 和 `22×12=264` soft tokens，均在 `2520/280` 上限内。

随机尺寸必须记录原始尺寸、实际渲染尺寸、缩放比例和视觉 token 数。Minecraft 相机监督值是 `[pitch, yaw]` 世界角度；普通图像 resize 不改变角度标签。只有从屏幕像素目标反推视角时，才根据实际 FOV 和渲染尺寸换算角度，再按动作编码合同换算 mouse count。

## 课程与快照门禁

| 课程特征 | 当前处理 |
|---|---|
| 静态方块、同维度、玩家稳定状态 | 可进入内存快照 `2+6` |
| 困难但可行的起点 | 保留；跑、游泳、准备补给均属于有效策略 |
| 缺少必要工具或材料 | `PREPARATION_REQUIRED`，生成准备课程 |
| 指标持续改善且方法正确 | `PROGRESSING`，增加有限预算并从稳定检查点继续 |
| 长时间无进展但无不可行证据 | `UNKNOWN`，分析观察与指标，不直接判失败 |
| 物理、时间或精度上有明确不可行证据 | `INFEASIBLE` |
| 普通实体依赖 | 当前拒绝公平快照 batch |
| 方块/流体计划刻依赖 | 当前拒绝公平快照 batch |
| 跨维度 | 当前拒绝公平快照 batch |

`tools/audits/codex_teacher_batch8.py` 是教师轨迹的真实执行入口。它证明执行、复位、截图和报告链路，不证明课程生成已经通用于采矿或长期任务。通用状态决策位于 `curriculum/runtime.py`，真正的观察驱动课程提议器仍需在正式模型接入时输出结构化目标、前置条件、可度量进展和快照能力要求。

## 正式开训前剩余门槛

| 优先级 | 必须完成的验证 | 通过条件 |
|---|---|---|
| P0 | 正式 trainer 接入联合 token loss | 单步真实模型反传；来源掩码、token 掩码与梯度审计通过 |
| P0 | on-policy 元数据 | 六条 policy 保存模型版本、采样配置和逐 token `old_logprob` |
| P0 | 实体快照 | 实体 UUID、位置、生命、AI、载具关系恢复且连续 8 次无重复 |
| P0 | 计划刻快照 | 方块和流体计划刻恢复次序、延迟与去重通过真实测试 |
| P0 | 跨维度 | 显式维度恢复完成；在此之前维度课程使用独立世界或进程 |
| P1 | 三类不同观察课程 | 至少采集、制作、长任务检查点各完成一个真实 `2+6` |
| P1 | 长任务续跑 | 真实完成 `PROGRESSING → 保存 → 扩预算 → 恢复 → 继续` |
| P1 | PRO 6000 压测 | 推理与训练异步重叠，记录 GPU 利用率、队列深度、过期 tick 和空动作率 |
| P2 | Ruff | 在完整开发环境运行并达到零错误 |

因此，本轮完成了可在无真实大模型本机完成的训练数据、梯度、课程状态、动作续算、静态快照和报告验证。整个项目尚未达到“所有课程可正式训练”的条件；上表 P0 项完成前，训练范围必须保持受限。
