# CraftGround TAP 闭环

## 数据流

```text
历史 RGB 帧
  -> 策略模型生成 TAP 动作块
  -> decode_action_sequence()
  -> action_tick_to_v2_action()
  -> CraftGround environment.step()
  -> 新 RGB 帧和 trajectory.json
```

一个分号分隔一个 50 ms tick。服务逐 tick 执行动作并保存观测，下一轮模型调用使用真实执行后的
图像。每条候选轨迹开始前，服务从同一个 JVM 内存快照恢复场景，并重新设置玩家位置和背包。

## 启动

```bash
python -m game_environment.closed_loop_server \
  --runtime /path/to/patched/craftground-runtime \
  --output runs/craftground-closed-loop \
  --host 127.0.0.1 \
  --port 18400 \
  --max-ticks 400 \
  --max-turns 10
```

## HTTP 契约

健康检查：

```http
GET /health
```

开始候选轨迹：

```http
POST /reset
Content-Type: application/json

{"trajectory_id": "T1"}
```

执行一个完整 TAP 动作块：

```http
POST /step
Content-Type: application/json

{
  "action_text": "<|action_start|> ; W ; Mouse -20 10 W ; MouseRight <|action_end|>",
  "model": {"model": "policy-name", "prompt_kind": "history_to_future_action"}
}
```

异步滚动执行使用两个接口。先提交至少 8 tick 的未来计划：

```http
POST /enqueue
Content-Type: application/json

{
  "plan_id": "turn-0001",
  "start_tick": 12,
  "action_text": "<|action_start|> ; W ; W ; W ; W ; W ; W ; W ; W <|action_end|>",
  "model": {"model": "policy-name"}
}
```

再由环境时钟持续推进：

```http
POST /advance
Content-Type: application/json

{"ticks": 1}
```

响应中的 `action_queue.should_replan` 指示是否应启动下一轮异步推理。触发剩余量为
`max(4, ceil(plan_ticks / 4))`：8 至 16 tick 的计划至少提前 4 tick，长计划在剩余四分之一时
续算。新计划只覆盖 `start_tick` 以后的旧队列；到达时已经过去的动作前缀被丢弃。
`/advance` 队列为空时仍推进世界，并释放所有按键、鼠标增量和滚轮动作。

`trajectory_id` 只能包含字母、数字、点、下划线和连字符。单个请求体上限为 1,000,000 字节。
服务会按 `max_ticks` 截断超出预算的动作，并在 `max_turns` 耗尽后拒绝新动作。

CraftGround V2 没有相对滚轮字段，闭环适配器会维护当前快捷栏槽位，并把 `Scroll N` 转换为
`hotbar.1` 至 `hotbar.9` 的绝对选择动作。每条轨迹从第 1 格开始；正数表示向上滚，槽位编号
递减，负数表示向下滚，槽位编号递增，两端循环。例如第 1 格执行 `Scroll 5` 后选中第 5 格。
同一 tick 不能同时使用 `Scroll` 和快捷栏数字键，因为 CraftGround V2 无法表达二者的执行顺序。
`/step` 状态和 `trajectory.json` 的每个已执行 chunk 都记录解析后的 `selected_hotbar`。

## 已验证结果

2026-07-31 的四轨迹闭环实验使用同一内存快照作为起点。T4 在第 9 次模型指令、tick 25 打开箱子
界面。该结果证明 TAP 动作文本、逐 tick V2 适配、真实 RGB 回灌和内存快照恢复可以组成完整
闭环。实验图片、原始模型回复和运行日志属于 `runs/` 生成物，不进入源码仓库。

## 状态边界

当前内存快照覆盖区域内方块、流体方块状态和方块实体 NBT。玩家位置与背包由每条轨迹开始前的
固定命令恢复。普通实体、玩家完整状态、方块计划 tick 和流体计划 tick 尚未纳入快照，因此当前
实现适用于状态边界受控的场景。
