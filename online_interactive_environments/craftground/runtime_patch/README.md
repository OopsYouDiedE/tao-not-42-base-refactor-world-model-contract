# CraftGround 内存快照扩展

该补丁面向 CraftGround 2.7.4 和 Minecraft 1.21。将 `MemorySnapshotStore.kt` 放入 runtime 的
`src/main/java/com/kyhsgeekcode/minecraftenv/`，并在 `MinecraftEnv.handleCommand` 开头调用：

```kotlin
if (MemorySnapshotStore.handle(command, client)) return true
```

补丁提供以下进程内命令：

```text
memorysnapshot save <id> <x1> <y1> <z1> <x2> <y2> <z2>
memorysnapshot load <id>
```

每个 CraftGround JVM 保存自己的快照。多个 JVM 必须在同一逻辑状态下以相同 ID 分别保存，随后
`ParallelRolloutRunner` 才能把该 ID 分配给不同 SubAgent。快照包含区域内方块、流体、方块实体
NBT、区域内普通实体，以及玩家位置、视角、速度、背包、生命、饥饿、经验和状态效果。加载时先
清除区域内当前普通实体，再放置快照实体，避免掉落物和生物在重复推演间残留。它不包含计划方块
tick、计划流体 tick 和客户端 GUI；需要可复现比较时应关闭随机 tick、昼夜、天气和生物生成。
