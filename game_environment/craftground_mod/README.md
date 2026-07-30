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

当前版本使用 Minecraft `StructureTemplate`，快照只保存在 JVM 内存中，能够恢复方块、流体方块状态和
方块实体 NBT。箱子内容、熔炉物品、燃烧计数和烹饪计数已经通过真实 CraftGround 验证。

当前版本不保存玩家、普通实体、方块计划刻和流体计划刻。正式相对优势训练仍需扩展这些状态域，不能把
当前通过结果解释为完整世界状态已经能够恢复。

2026-07-31 的真实单 JVM 测试中，`9×6×9` 区域保存耗时为 `33.78 ms`，恢复耗时为 `28.34 ms`。
耗时包含两次 CraftGround IPC 同步 tick。状态恢复六项断言全部通过。多环境并行广播已经通过单元测试，
尚未在本机同时启动八个 Minecraft 客户端做真实性能测试。

真实测试命令：

```bash
PYTHONPATH=. xvfb-run -a .venv/bin/python \
  tools/verify_craftground_memory_snapshot.py \
  --runtime /path/to/patched/craftground-runtime \
  --output runs/craftground-memory-snapshot-report.json
```
