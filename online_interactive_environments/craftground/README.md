# CraftGround 并行推演

本项目把 CraftGround 上游的 `ActionSpaceVersion.V2_MINERL_HUMAN` 称为
`keyboard_and_mouse_only` 后端。该名称描述当前设备范围，不代表标准输入动作协议版本。

## 环境入口

项目直接维护 [OopsYouDiedE/CraftGround](https://github.com/OopsYouDiedE/CraftGround/tree/tao-maintained)
的 `tao-maintained` 分支。核心包与 mc121 runtime 当前共同锁定到提交
`94d211204757fa8ba0f6182a72b81071e80c3fd5`，不得改回 PyPI 范围依赖或只锁分支名。
只有上游通过公开 API 提供等价可调能力并完成真实环境验证后，才重新评估迁回上游。

默认入口从已安装的维护版 `craftground-runtime-mc121` 创建构建模板，校验维护版能力并完成首次
Gradle 构建，不再修改第三方包源码。每次创建环境时再从模板复制独立可写实例目录，使
`.gradle/`、`build/`、`run/saves/`、`run/logs/` 和 `options.txt` 不在 JVM 之间共享：

```python
from online_interactive_environments.craftground import create_environment

environment = create_environment()
observation, info = environment.reset(options={"fast_reset": False})
```

模板缓存按 runtime 版本和源码摘要隔离，实例缓存按 `instance_id` 隔离。源码未变化时不会重复构建。
传入 `runtime_path` 表示直接使用一个已经准备好的 runtime，此时入口不会修改它：

```python
environment = create_environment(runtime_path="C:/craftground/maintained-runtime")
```

显式 `runtime_path` 由调用方负责可写目录隔离。入口默认 `find_free_port=False`；端口已占用时直接
失败，不静默改用其他端口。

## IPC 传输

入口默认使用共享内存 IPC，观察不经 socket 序列化，也不触发 SocketIPC 那个不分端口的全局 java
进程扫描。

维护分支直接修复共享内存重复初始化、动作段 0 字节定容、动作越界写入、重复销毁和 POSIX 观察段
扩容问题。主项目不再 monkeypatch `BoostIPC`，也不在运行时替换 Kotlin 或 C++ 源码。

## 依赖边界

| 层级 | 维护版约束或上游现状 |
| --- | --- |
| 主项目 pin | 两个 CraftGround 包使用同一 40 位 Git 提交 |
| Python | 主项目要求 Python `>=3.11`；runtime 上游声明 `>=3.9`，核心包未声明下限 |
| Java/Minecraft 1.21 | JDK 21、Gradle 8.8、Kotlin 2.0.0、Fabric Loader 0.15.11 |
| Native | CMake 实际要求 `>=3.28`，并需要 JNI、OpenGL；非 macOS 需要 GLEW |
| 可选渲染 | PNG `>=1.6`、CUDA Toolkit；CUDA 不属于本轮验证范围 |
| Python 间接依赖 | 上游多数未设版本边界，由本项目环境锁定与真实验收控制 |

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
