# 旧代码符号迁移账本

本文是旧代码符号迁移的正式审计记录。迁移按公开 `class` / `def` 筛选，
实现压缩到聚合模块；轨迹、图片、运行结果、UI 和旧命令外壳不迁入。

| 旧符号组 | 决策 | 当前位置或理由 |
| --- | --- | --- |
| MineStudio 动作编码 | 迁入并统一协议 | 数据直接转换为 `standard-input-action/v1` 的 `ActionSequence`，不保留中间领域类型 |
| `MineStudioDataset`, `load` | 迁入并修正 | `external_dataset_loaders_and_protocol_adapters/minestudio.py`；重依赖延迟导入，删除破坏性清理参数，修正 `update_index` 名称 |
| `EpisodeIdentity`, `SplitResult`, split 函数 | 迁入并合并 | `behavior_cloning_dataset_converters/dataset_conversion.py` |
| SFT question/response 格式函数 | 迁入并简化 | `behavior_cloning_dataset_converters/dataset_conversion.py`；移除旧动作解码路径耦合 |
| HDF5 load、流式 BC dataset | 已按新协议恢复 | `behavior_cloning_training/dataset.py`；统一产出标准输入动作协议 v1 conversation，不恢复旧 TAP 类型 |
| 视觉 BC LoRA 训练 | 已按新职责恢复 | `behavior_cloning_training/train.py`；模型重依赖延迟导入 |
| 行为克隆、相对优势、联合目标、视觉几何 | 迁入并合并 | `relative_advantage_comparison_training/objectives.py`；PyTorch 延迟导入 |
| `RolloutSample` 与 2+6 校验函数 | 迁入并合并 | `relative_advantage_comparison_training/rollouts.py` |
| on-policy rollout 与 clipped RLHF | 已按新职责恢复 | `relative_advantage_comparison_training/policy_rollout.py`、`relative_advantage_comparison_training/train_policy.py` |
| `ReviewCandidate` 与审核奖励函数 | 删除 | 迁移后未接入模型判断审核生产流程，只有自测消费者 |
| 课程状态、快照准入、继续决策、分层抽样 | 删除 | 迁移后未接入环境验证生产流程，只有包级重导出 |
| 旧 CraftGround runtime、动作调度、内存快照 | 由当前实现替代 | `online_interactive_environments/` 已存在职责更清楚的新实现，不重复定义协议 |
| 旧 Codex client、teacher pipeline | 由当前实现替代 | `online_environment_interaction_agents/` 已有教师执行链路 |
| Gradio 审核 UI、HTTP server、GPU watchdog、凭证导出、旧 CLI `main` | 删除 | 属于工具外壳、临时运行设施或不合适的职责耦合 |
| `runs/`, `artifacts/`, 示例图片、审核轨迹 JSON/Markdown | 删除 | 纯产物，不属于可迁移代码 |
