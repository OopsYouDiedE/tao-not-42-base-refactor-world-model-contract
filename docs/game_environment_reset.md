# CraftGround 内存快照恢复

## 当前实现

当前管线在常驻 CraftGround JVM 内使用 Minecraft `StructureTemplate` 保存受控区域。Python 侧的
`MemorySnapshotCoordinator` 向一个或多个环境发送同名快照命令，并在恢复时并行等待同步 tick。

```python
from game_environment import MemorySnapshotCoordinator, SnapshotRegion

coordinator = MemorySnapshotCoordinator(environments)
snapshot = coordinator.capture_all(
    "episode-42-step-180",
    SnapshotRegion((0, 63, 0), (8, 68, 8)),
)
timings = coordinator.reset_all(snapshot)
```

每个 JVM 保存自己的快照。多环境使用相同快照 ID，不共享 JVM 内存对象。恢复墙钟时间由最慢的
环境决定，`ResetTimings` 同时记录总墙钟和各 worker 耗时。

## 状态范围

| 状态 | 当前支持 |
|---|---|
| 普通方块 | 是 |
| 当前流体方块状态 | 是 |
| 箱子、熔炉等方块实体 NBT | 是 |
| 玩家位置、视角、背包 | 由闭环开始命令固定恢复 |
| 普通实体与生物 AI | 否 |
| 玩家完整属性 | 否 |
| 方块与流体计划 tick | 否 |

受控场景之外的相对优势分支需要先扩展 Kotlin 快照协议。磁盘世界目录复制实现已经退出当前管线，
因为它要求关闭 JVM 并重新启动环境，无法满足当前的低延迟闭环要求。

## 验证

Kotlin 扩展的安装方式见 `game_environment/craftground_mod/README.md`。安装后执行：

```bash
python -m game_environment.verify_memory_snapshot \
  --runtime /path/to/patched/craftground-runtime \
  --output runs/craftground-memory-snapshot-report.json
```

验证器会保存石头、水、箱子物品和熔炉状态，随后主动破坏场景并恢复快照。全部状态断言与一秒内
恢复条件同时通过时，进程以状态码 0 结束。
