# CraftGround 并行推演

本项目把 CraftGround 上游的 `ActionSpaceVersion.V2_MINERL_HUMAN` 称为
`keyboard_and_mouse_only` 后端。该名称描述当前设备范围，不代表标准输入动作协议版本。

## 环境入口

默认入口会从已安装的 `craftground-runtime-mc121` 创建构建模板，自动注入内存快照补丁并完成首次
Gradle 构建。每次创建环境时再从模板复制独立可写实例目录，使 `.gradle/`、`build/`、`run/saves/`、
`run/logs/` 和 `options.txt` 不在 JVM 之间共享：

```python
from online_interactive_environments.craftground import create_environment

environment = create_environment()
observation, info = environment.reset(options={"fast_reset": False})
```

模板缓存按 runtime 版本隔离，实例缓存按 `instance_id` 隔离。补丁内容未变化时不会重复构建。传入 `runtime_path` 表示直接使用一个
已经准备好的 runtime，此时入口不会修改它：

```python
environment = create_environment(runtime_path="C:/craftground/patched-runtime")
```

显式 `runtime_path` 由调用方负责可写目录隔离。入口默认 `find_free_port=False`；端口已占用时直接
失败，不静默改用其他端口。

## IPC 传输

入口默认使用共享内存 IPC，观察不经 socket 序列化，也不触发 SocketIPC 那个不分端口的全局 java
进程扫描。

上游共享内存路径存在一处重复初始化：`CraftGroundEnvironment.__init__` 已经通过 `BoostIPC` 创建
`/craftground_<port>_p2j` 与 `_j2p` 并写入初始环境消息，随后 `reset()` 内的 `ensure_alive()` 会对
同一端口再构造一个 `BoostIPC`，命中 native 层的 “already exists” 检查而失败。先 `destroy()` 再让
它重建同样不可行：destroy 之后 native 模块对该段名的映射失效，读侧拿到坏 fd 并返回空字节，Python
侧表现为 `cannot reshape array of size 0`。

`enable_shared_memory_reuse()` 按端口缓存首次初始化结果，重复构造直接复用已有段名，不再调用
native 初始化，也不销毁正在使用的段。同时把隐式析构改为显式 `release()`，由 `create_environment`
绑定到环境 `close()` 之后，避免被丢弃的旧实例 unlink 掉新实例仍在使用的段。

该包负责把 CraftGround 常驻实例分配给多个 SubAgent。每个实例在同一时刻只归一个 SubAgent 使用。

```python
coordinator = MemorySnapshotCoordinator(environments)
snapshot = coordinator.capture_all("root-state", region)
runner = ParallelRolloutRunner(coordinator, max_workers=8)
results = runner.run(requests, wait_timeout=30.0)
```

## 固定存档并行启动

四个 CraftGround JVM 不共享同一个可写世界目录。先在 WSL 中创建只读基准存档，再由环境入口把它复制到四个独立 runtime：

在 Windows 项目根目录执行以下一条命令，可以自动创建基准存档、运行四条轨迹、完成教师测评，并在时间戳运行目录中生成 `REPORT.md`：

```powershell
wsl.exe -d Ubuntu-24.04 -- /home/zznzz/.cache/tao/venvs/four-teacher/bin/python -m environment_validation_tools.run_complete_teacher_evaluation
```

需要指定目录或复用已有基准存档时：

```powershell
wsl.exe -d Ubuntu-24.04 -- /home/zznzz/.cache/tao/venvs/four-teacher/bin/python -m environment_validation_tools.run_complete_teacher_evaluation --output runs/complete-teacher-evaluation --baseline-world runs/four-teacher-baseline/baseline-world
```

底层分步命令如下：

```bash
python -m environment_validation_tools.create_craftground_baseline_world \
  --output runs/four-teacher-baseline \
  --port 19510

python -m environment_validation_tools.run_four_teacher_trajectories \
  --output runs/four-teacher-fixed-world \
  --baseline-world runs/four-teacher-baseline/baseline-world \
  --socket-ipc \
  --port-base 19800 \
  --backend codex-cli
```

入口在 JVM 启动前校验四份存档源哈希相同、实例路径互不重复。世界快照加载后还会通过 `clear @p`、`tp @p` 和状态回读恢复玩家起点；状态不一致时最多重新提交五次，仍不一致则停止运行。

`capture_all` 会在每个 JVM 内保存同名快照。每项推演取得独占环境后，先加载请求指定的快照，再调用
请求中的 `simulate(environment, payload)`。线程池并行驱动多个独立 JVM，因此环境仿真和 IPC 等待可以
同时进行。请求数量超过环境槽位时，请求在 `EnvironmentPool` 中等待；`wait_timeout=None` 表示持续
等待，给定秒数后仍无槽位则抛出 `EnvironmentPoolTimeout`。推演成功或抛出异常时租约都会归还。

快照是 JVM 本地对象。同一 ID 必须预先广播保存到所有可能接收该任务的环境，不能把一个 JVM 中的
`StructureTemplate` 直接传给另一个 JVM。

可比较推演还要求各 JVM 在保存同名快照前处于同一逻辑状态。四分支验证入口使用固定种子，冻结昼夜、
天气、生物生成和随机 tick，清除普通实体，并比对玩家位置、视角、生命、背包和视线方块。快照区域
根据实际玩家坐标计算；保存后入口主动扰动各环境，再倒档并比对状态，以验证实际恢复路径。
