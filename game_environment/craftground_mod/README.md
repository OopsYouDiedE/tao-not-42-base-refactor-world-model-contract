# CraftGround 内存快照扩展

`MemorySnapshotStore.kt` 为 CraftGround 2.7.4 / Minecraft 1.21 提供单进程内存结构快照。

将文件放入 CraftGround runtime 的 `src/main/java/com/kyhsgeekcode/minecraftenv/`，并在
`MinecraftEnv.handleCommand` 的开头加入：

```kotlin
if (MemorySnapshotStore.handle(command, client)) return true
```

扩展提供两个 IPC 命令：

```text
memorysnapshot save <id> <x1> <y1> <z1> <x2> <y2> <z2>
memorysnapshot load <id>
```

Python 调用方使用 `MemorySnapshotCoordinator`，无需直接拼接命令：

```python
from game_environment import MemorySnapshotCoordinator, SnapshotRegion

coordinator = MemorySnapshotCoordinator(environments)

# 所有常驻环境在相同逻辑状态下，以同一 ID 保存各自的 JVM 内存快照。
snapshot = coordinator.capture_all(
    "episode-42-step-180",
    SnapshotRegion((0, 63, 0), (8, 68, 8)),
)

# 后续只传快照句柄，即可并行恢复全部环境。
timings = coordinator.reset_all(snapshot)
assert timings.wall_ms < 1000
```

每个独立 JVM 都有自己的内存快照表。`capture_all` 会向全部环境广播同一个快照 ID，`reset_all` 使用线程池
并行广播恢复命令，因此总墙钟时间由最慢 worker 决定。一个 JVM 中创建的 `StructureTemplate` 不会跨进程
自动共享。

当前版本使用 Minecraft `StructureTemplate` 保存方块、流体和方块实体 NBT，同时使用玩家原生 NBT
保存位置、速度、视角、背包、选中槽、生命、饥饿、经验和状态效果。恢复发生在同一个服务器线程中，
随后通过网络处理器把位置和视角同步到客户端。

当前版本仍不保存普通实体、方块计划刻和流体计划刻，也不恢复客户端 GUI。调用方必须在候选结束时关闭
GUI，并在恢复后执行移动探针。正式相对优势训练不能把当前实现解释为完整世界状态已经能够恢复。

2026-07-31 的真实单 JVM 测试中，`9×6×9` 区域保存耗时为 `33.78 ms`，恢复耗时为 `28.34 ms`。
耗时包含两次 CraftGround IPC 同步 tick。状态恢复六项断言全部通过。多环境并行广播已经通过单元测试，
尚未在本机同时启动八个 Minecraft 客户端做真实性能测试。

真实测试命令：

```bash
PYTHONPATH=. xvfb-run -a .venv/bin/python \
  game_environment/verify_memory_snapshot.py \
  --runtime /path/to/patched/craftground-runtime \
  --output runs/craftground-memory-snapshot-report.json
```

玩家状态连续恢复测试：

```bash
PYTHONPATH=. .venv/bin/python -m game_environment.verify_player_snapshot \
  --runtime /path/to/patched/craftground-runtime \
  --output runs/player-snapshot-eight-restore/report.json
```

2026-07-31 的真实 CraftGround 测试连续恢复 8 次，位置、视角、背包、生命、饥饿、饱和度、经验和状态效果全部一致。
