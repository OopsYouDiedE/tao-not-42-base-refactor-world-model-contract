# CraftGround 世界存档与动作重放

## 1. 能力边界

CraftGround 当前没有暴露运行中 JVM 的内存快照接口。可恢复的边界是 Minecraft 已落盘的完整世界目录，
因此本项目采用“不可变存档 + 可丢弃工作副本 + 冷启动”的机制，不把固定 seed 或初始化命令称为读档。

保存时先向服务器发送 `save-all flush`，等待一个环境 tick，再归档包含 `level.dat`、区块、实体和玩家数据的
完整世界目录。`session.lock` 属于进程锁，不进入快照。每个快照记录逐文件 SHA-256，创建后禁止原地覆盖。

加载时先关闭旧环境，把快照复制到目标 MinecraftEnv 独占的 `run/saves`，然后通过
`level_display_name_to_play` 冷启动新 JVM。工作副本可以被 CraftGround 在退出时删除，不影响不可变存档。

## 2. 重放契约

轨迹记录 CraftGround V2 动作的全部键，包括按键、九个快捷栏和相机俯仰/水平增量。缺键或多键直接报错，
避免离散动作编号随着动作表修改而失去原始语义。

保存点和每个动作后的完整观测会生成状态指纹。指纹覆盖位置、朝向、生命、饥饿、经验、世界时间、天气、
选中栏及逐槽库存，不包含 RGB，避免渲染噪声被误判为物理分叉。读档后从 tick 0 开始逐步比较，首次不一致
会抛出 `ReplayDivergence`，而不是继续产生表面可播放、实际已偏离的轨迹。

该指纹是在线一致性检查，不替代磁盘世界哈希。磁盘哈希保证存档文件未变化；在线指纹保证玩家与任务相关
状态能够复现。若实验依赖特定方块或实体，应把对应字段加入完整观测和指纹契约。

## 3. 代码边界

- `world_snapshot.py` 负责不可变归档、完整性校验和工作副本恢复。
- `replay.py` 负责 V2 动作序列化、状态指纹、运行中保存和逐 tick 重放。
- `replay_runtime.py` 负责真实 CraftGround JVM 的冷启动装配。
- 离线协议替身只存在于 `tests/integration/`，生产代码不包含 CraftGround mock。

并行训练时，每个环境必须使用独立 MinecraftEnv 目录和端口。运行中的环境不得与恢复操作共享同一个
`run/saves`，否则文件锁、退出清理和工作副本替换会互相干扰。

## 4. 调用流程

1. 为环境确定独立 MinecraftEnv 路径和世界显示名。
2. 用 `capture_running_world` 执行 flush 并取得起点快照及 flush 后完整观测。
3. 用 `TrajectoryRecorder` 从该返回观测开始记录每个完整 V2 动作及动作后观测，不能复用 flush 前观测。
4. 关闭旧环境；不得复制仍由 JVM 写入的工作世界。
5. 用 `CraftGroundReplayRuntime.load_and_replay` 恢复工作副本、冷启动并校验重放。

快照、轨迹与训练产物应放在 `runs/`，不进入版本库。
