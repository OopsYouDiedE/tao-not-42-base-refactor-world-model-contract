# 正式项目动作协议 Prompt

本实验每轮使用 `datasets/minestudio_finetune/sft_protocol.py` 中的
`history_to_future_action` Prompt：

```text
The images are past Minecraft observations in chronological order. Infer one reasonable future
action block. Choose a suitable number of 50 ms ticks from the visible action type and required
duration instead of waiting for a supplied target length. Keep brief actions short; sustained
movement, mining, attacking, drawing, eating, or continuous use may last up to 60 ticks. Omit
unsupported 1-2 pixel camera jitter and do not invent GUI clicks or auxiliary keys without visual
evidence.

Action format example for a 3-tick block:
"<|action_start|> ; W ; Mouse 4 -2 W ; W <|action_end|>".
Each JSON array item must be one string action block; do not return nested tick arrays.

Output the complete executable JSON action array first. Then start a new line with "Reason:" and
briefly explain the visual evidence, intent, and duration choice. The action array must remain
independently parseable because generation may stop after it.
```

每轮还提供：按时间排列的真实历史 RGB、当前剩余 tick、剩余模型指令数和此前模型原始输出。
没有加入人工导航策略。每轮实际 Prompt 和原始模型输出保存在对应 `trajectory.json`。
