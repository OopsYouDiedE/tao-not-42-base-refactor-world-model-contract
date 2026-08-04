# 云端模型轨迹相对优势比较流程

本文记录「轨迹完全由云端大模型生成、随后统一评估相对优势」的运行合同、启动参数和已知故障处理。
运行入口为 `scripts/run_wsl_cloud_relative_advantage.sh`，它封装
`environment_validation_tools.run_four_teacher_trajectories`。

## 流程

所有 arm 使用同一云端模型和同一提示词，构成同策略多分支样本。单个 arm 内的循环是
「观察 → 模型生成一段协议动作 → 环境真实执行 tick → 新观察」，模型生成轮次上限由
`--max-generations` 控制。全部 arm 结束后统一执行：

| 顺序 | 模块 | 输出 |
| --- | --- | --- |
| 1 | `interaction_trajectory_review_agents.review_trajectory` | 逐轨迹合同、预算与任务结果审核 |
| 2 | `relative_advantage_comparison_training.build_comparison_group` | 以组内均值为基线的相对优势与排名 |
| 3 | `model_judgment_review_agents.review_comparison` | 比较均值、排序和选择结论复核 |

相对优势以组内均值为基线，因此同一组内相对优势之和为零（受 6 位小数舍入影响）。

## 运行环境

Windows 上必须使用 WSL 2 Ubuntu-24.04。真实 CraftGround 只在 Linux 运行，`run` 默认
`enforce_wsl=True`。

## 凭据与 wire 协议

入口不读取 `~/.codex/auth.json` 或任何 CLI 私有凭据文件，`TEACHER_API_KEY` 必须由调用方显式导出。

`gpt-5.6` 系列（`sol`、`terra`、`luna`）只支持 OpenAI **responses** 协议。使用 chat completions
调用会得到 HTTP 400 `protocol_not_supported`，因此必须设置 `TEACHER_WIRE_API=responses`。
`OpenAICompatibleConfig.from_environment()` 会读取该变量与 `TEACHER_TIMEOUT_SECONDS`。

同一网关下不同密钥可能属于不同分组。分组缺少对应渠道时返回 HTTP 503 `model_not_found`
并提示「无可用渠道（distributor）」；这是密钥分组问题，不表示模型不存在。

## 环境槽位与内存

单个 Minecraft 客户端约需 1.5 GB 堆（`CRAFTGROUND_JVM_MAX_MEMORY=1500m`）。arm 数与并行环境数
通过 `--environment-count` 解耦：超出槽位的 arm 由 `ParallelRolloutRunner` 在环境池外排队，不会
同时占用内存。

在 10 GB 内存的 WSL 上，8 arm 配 4 槽位出现过其中一个客户端 `Minecraft process failed to start`
（该实例连 `run/logs/` 都未创建）。降到 2 槽位后 8 arm 全部完成。内存紧张时优先调小
`--environment-count`，而不是减少 arm 数。

首次使用某个 runtime 实例会触发 Gradle 构建补丁，耗时可达数分钟，属正常现象。

## 已验证运行

2026-08-04 在 WSL 2 Ubuntu-24.04 上完成一次真实运行：`gpt-5.6-terra`、`openai-api` 后端、
responses wire、8 arm、每 arm 最多 10 轮生成、单 arm 128 tick 预算、2 环境槽位、socket IPC，
复用基准存档 `a6fe0d9973f1986434037dd42541983bffc740bdd988dc273bfe23b4ececdcfe`。

结果：8 条轨迹全部完成，共享起点指纹一致，快照恢复探针通过，比较复核 `valid=true`，排名覆盖
1/2/3 三档，相对优势之和为 `1e-06`（舍入残差）。全部生成记录的 telemetry 均标记
`provider=openai-compatible`、`model=gpt-5.6-terra`。8 条轨迹的任务成功均为否。

## 已知评分缺陷

`review_trajectory` 的质量分为
`(1.0 if task_success else 0.0) + 0.25 * max(0, 1 - executed_ticks / budget)`，issue 惩罚为
`min(0.5, 0.1 * len(issues))`。在任务全部失败的组内，效率项成为唯一区分度来源，因此**早退轨迹
得分最高**。

本次运行中 T06 因协议错误在第 6 tick 中止，效率项得 `0.25 * 0.953 = 0.238`，扣除单个 issue 的
`0.1` 后仍为 `0.138`，高于跑满 128 tick 且合同审核通过的轨迹（`0.0`），从而被 `review_comparison`
选为最佳轨迹。

这意味着当前评分把「尽早失败」当作优势。该缺陷属于评分函数设计，不是本次运行的执行故障：本次
比较流程本身按合同完成。修复方向需要单独决策，例如任务失败时不给效率奖励、或让合同审核失败的
轨迹不参与最佳选择。修复前不应把该评分用于驱动相对优势训练。
