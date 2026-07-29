# MineStudio 动作测试成果

本目录保存动作选择测试的稳定副本，不依赖 `runs/` 临时目录。

## 测试版本

| 版本 | 内容 | 主要结果 |
|---|---|---|
| `action_choice_easy_100` | Easy 四选一动作匹配 | SubAgent 结果副本 |
| `action_choice_counterfactual_100` | 旧反事实方向/时序测试 | 存在明显结构泄漏 |
| `action_choice_magnitude_100` | 动作幅度程度测试 | 无图中心基线约 24% |
| `action_choice_wasd_mouse_100` | WASD ±4/±8 帧、鼠标局部乘除 | SubAgent 54/100；无图中心基线 27/100 |

每个版本目录包含 `manifest.json`、公开题目 `questions.jsonl`，以及已有的 SubAgent 预测文件。封存答案仍保留在原实验目录中，避免公开题目副本直接泄漏答案。

## 代码与测试

- [生成器共享基础操作](../../minestudio/action_benchmark_common.py)
- [Easy/反事实/幅度测试生成器](../../minestudio/action_choice_benchmark.py)
- [WASD 与鼠标幅度生成器](../../minestudio/action_magnitude_benchmark.py)
- [意图保持机械切分器](../../minestudio/mechanical_segmentation_probe.py)
- [动作选择测试](../../../tests/unit/test_action_choice_benchmark.py)
- [幅度测试](../../../tests/unit/test_action_magnitude_benchmark.py)
- [切分测试](../../../tests/unit/test_mechanical_segmentation_probe.py)

## 50 条图像抽样复核

[机械切分抽样报告](../../../artifacts/mechanical_segmentation/intent_validation_50_samples/agent_visual_validation.md)

判定标准是：切分后的前后图像是否展现出一个完整、连贯、未被切断的动作意图，而不是只检查动作标签是否相同。

| 指标 | 结果 |
|---|---:|
| 抽样数 | 50 |
| 完整动作意图 | 40 |
| 切分成功率 | 80% |
| 不完整或无法判断 | 10 |
| 平均置信度 | 约 0.74 |

抽样图像和原始动作摘要位于 [`artifacts/mechanical_segmentation/intent_validation_50_samples/`](../../../artifacts/mechanical_segmentation/intent_validation_50_samples/)。
