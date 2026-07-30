# 游戏环境快速重置与状态回退

## 结论

Minecraft/CraftGround 没有公开运行中 JVM 的完整内存快照接口。能够复现玩家、实体、区块、库存和世界时间的可靠边界，是游戏已落盘的完整世界目录。因此，相对优势训练采用不可变基准快照和可丢弃工作副本。

| 阶段 | 操作 | 一致性要求 |
|---|---|---|
| 建立分支点 | 发送 `save-all flush`，再等待一个环境 tick | 后续轨迹从 flush 后观测开始 |
| 保存 | 调用 `WorldSnapshotStore.capture` | 复制完整世界，排除 `session.lock`，记录逐文件 SHA-256 |
| 执行候选 A | 在独占 Minecraft 实例中 rollout | 保存动作、奖励和关键状态指纹 |
| 回退 | 关闭实例，调用 `restore` 替换工作副本 | 不复制仍由 JVM 写入的目录 |
| 执行候选 B | 从恢复后的世界冷启动 | 首帧状态指纹必须等于分支点指纹 |
| 计算优势 | 比较相同分支点上的候选回报 | 每个候选使用相同快照 ID |

## 调用示例

```python
from game_environment import WorldSnapshotStore, discover_world_dir

store = WorldSnapshotStore("runs/world-snapshots")

# 先通过环境命令接口执行 `save-all flush` 并等待一个 tick。
world = discover_world_dir("MinecraftEnv/run/saves")
store.capture(
    "episode-0042-step-0180",
    world,
    display_name="Relative Advantage World",
    state_digest=full_observation_digest,
)

# rollout 结束后先关闭 Minecraft/JVM，再生成新的工作副本。
restored_world, manifest = store.restore(
    "episode-0042-step-0180",
    "worker-02/MinecraftEnv/run/saves",
    slot_name="relative-advantage-world",
    replace=True,
)
```

## 快速化路径

| 方案 | 完整性 | 适用范围 |
|---|---|---|
| 普通目录复制 | 完整 | 默认方案，跨平台且容易审计 |
| Linux Btrfs/XFS reflink 或 ZFS clone | 完整 | 大世界和高频分支，复制成本接近元数据操作 |
| tmpfs/RAM disk 工作副本 | 完整 | 世界可装入内存且机器内存充足 |
| Minecraft `/clone`、传送、清库存 | 不完整 | 只适合状态字段受严格约束的任务 |
| 固定 seed 后重新生成 | 不完整 | 只复现初始地形，不复现运行中状态 |

并行 worker 必须使用独立的 MinecraftEnv 目录、`run/saves` 和端口。快照目录保持只读语义，游戏只能写入恢复出的工作副本。完整观测指纹至少覆盖位置、朝向、生命、饥饿、经验、世界时间、天气、选中栏和逐槽库存；任务依赖实体或特定方块时，应把对应状态加入指纹。

## 已验证的内存恢复

CraftGround 2.7.4 / Minecraft 1.21 的真实进程测试已经验证 `StructureTemplate` 内存快照能够恢复石头、
水方块、箱子两槽物品、熔炉两槽物品、燃烧计数和烹饪计数。测试过程在同一 JVM 内完成保存、命令变异和
恢复，没有关闭 Minecraft，也没有读取世界存档。

当前实现尚未覆盖普通实体、玩家数据、方块计划刻和流体计划刻。流水的当前方块位置已经恢复，未来传播
队列尚未验证。正式训练接入前必须补齐这些状态并运行多 worker 的长时间一致性测试。

真实测试测得 `9×6×9` 区域的内存保存耗时为 `33.78 ms`，一键恢复墙钟耗时为 `28.34 ms`，均包含
CraftGround IPC 同步。Python 侧使用 `MemorySnapshotCoordinator.capture_all` 为全部常驻环境建立同名
快照，随后使用 `reset_all(snapshot)` 只传快照句柄并行恢复全部环境。
