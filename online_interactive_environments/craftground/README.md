# CraftGround 并行推演

本项目把 CraftGround 上游的 `ActionSpaceVersion.V2_MINERL_HUMAN` 称为
`keyboard_and_mouse_only` 后端。该名称描述当前设备范围，不代表标准输入动作协议版本。

## 环境入口

项目直接维护 [OopsYouDiedE/CraftGround](https://github.com/OopsYouDiedE/CraftGround/tree/tao-maintained)
的 `tao-maintained` 分支。核心包与 mc121 runtime 当前共同锁定到提交
`ac71d4ef6fb12b994d35b36f8eec518aa3a307e7`，不得改回 PyPI 范围依赖或只锁分支名。
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

`create_environment(screen_encoding_mode="raw")` 返回 NumPy RGB 帧；设置
`screen_encoding_mode="zerocopy_torch"` 后，OpenGL 颜色纹理通过设备到设备复制写入 CUDA IPC
共享 RGBA 缓冲区，Python 内部张量是该缓冲区的 live view。公开 RGB 返回会执行逐帧 `clone()`、
去 alpha 和垂直翻转，但仍位于 CUDA；因此这里的 zero-copy 边界是 JVM 到 Python 的 GPU 传输，
不表示最终 RGB 张量完全不发生 GPU 内复制。

## CUDA Linux 验收

真实 GPU 渲染需要 JDK 21、CUDA Toolkit，以及 OpenGL/Xorg 开发包。在 Ubuntu 上安装：

```bash
apt-get update
apt-get install -y openjdk-21-jdk libglew-dev libgl1-mesa-dev libglu1-mesa-dev \
  libglfw3-dev xorg-dev ninja-build mesa-utils xserver-xorg-core x11-xserver-utils pciutils
```

无桌面的 NVIDIA 服务器必须先启动 NVIDIA Xorg。先用 `nvidia-xconfig --query-gpu-info` 确认 GPU
BusID，再创建以下配置；本次 Tesla T4 环境的 BusID 为 `PCI:0:4:0`：

```text
Section "ServerLayout"
    Identifier "Layout0"
    Screen 0 "Screen0"
EndSection
Section "Device"
    Identifier "Device0"
    Driver "nvidia"
    BusID "PCI:0:4:0"
EndSection
Section "Screen"
    Identifier "Screen0"
    Device "Device0"
    DefaultDepth 24
    Option "AllowEmptyInitialConfiguration" "True"
EndSection
```

假设保存为 `/tmp/craftground-xorg.conf`，启动并验证：

```bash
Xorg :1 -config /tmp/craftground-xorg.conf \
  -modulepath /usr/lib64-nvidia/xorg/modules,/usr/lib/xorg/modules \
  -nolisten tcp -noreset -logfile /tmp/craftground-xorg.log

export DISPLAY=:1
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/cuda/lib64
glxinfo -B
```

`glxinfo -B` 必须同时显示 `direct rendering: Yes`、vendor `NVIDIA Corporation` 和实际 GPU
renderer；出现 `llvmpipe` 代表软件渲染，不算通过。不要同时设置 `UseDisplayDevice=None` 与
`Virtual`，NVIDIA 驱动会拒绝该 Screen 配置。

使用正式入口依次执行真实 RAW、共享内存和 `ZEROCOPY_TORCH` 验收：

```bash
python -m online_interactive_environments.craftground.validate_cuda_rendering \
  --output runs/craftground_cuda_validation
```

命令要求 `use_shared_memory=True`，实际启动 Minecraft、执行 reset 与多步动作，并验证 CUDA
tensor shape、dtype、device、CUDA IPC handle、live view data pointer 和帧变化。截图与
`report.json` 写入指定的 `runs/` 子目录，不提交到 Git。

## 依赖边界

| 层级 | 维护版约束或上游现状 |
| --- | --- |
| 主项目 pin | 两个 CraftGround 包使用同一 40 位 Git 提交 |
| Python | 主项目要求 Python `>=3.11`；runtime 上游声明 `>=3.9`，核心包未声明下限 |
| Java/Minecraft 1.21 | JDK 21、Gradle 8.8、Kotlin 2.0.0、Fabric Loader 0.15.11 |
| Native | CMake 实际要求 `>=3.28`，并需要 JNI、OpenGL；非 macOS 需要 GLEW |
| 可选渲染 | PNG `>=1.6`、CUDA Toolkit 12.8；RAW 与 CUDA IPC zero-copy 已在 Tesla T4 验证 |
| Python 间接依赖 | 上游多数未设版本边界，由本项目环境锁定与真实验收控制 |

## 控制内核

`EnvironmentKernel` 是唯一持有 CraftGround JVM 句柄的对象。装配、倒档、槽位调度和换基准都在它内部
完成，调用方不再自己拼装 runtime 目录、快照协调器和环境池：

```python
from online_interactive_environments.craftground import EnvironmentKernel

with EnvironmentKernel.launch(slots=4, port_base=19800, baseline_world="runs/baseline-world") as kernel:
    kernel.capture("root-state", region=region, as_root=True)
    results = kernel.rollout(requests, wait_timeout=30.0)
    kernel.reset()
```

`launch` 依次准备构建模板、逐槽位复制独立实例目录、安装基准存档并启动 JVM；任一槽位失败时已创建的
JVM 全部关闭，不留下悬挂进程。`close()` 幂等，内核本身是上下文管理器。

内核对外只暴露三类调用，与三种意图一一对应：

| 意图 | 调用 |
| --- | --- |
| 操控 | `lease()` / `handles()` 取得 `EnvironmentHandle`，再 `apply(tick)` |
| 重置 | `capture(id, region=..., as_root=True)` 保存根快照，`reset()` 或 `handle.reset_to()` 倒档 |
| 换基准 | `capture(..., as_root=True)` 覆盖根快照，或 `rebase(baseline_world)` 重建全部槽位 |

`rebase` 的两条路径代价不同。覆盖根快照不重启 JVM，但只影响快照区域；`rebase` 复用同一批端口，因此
先关闭当前内核再重建，返回一个新内核对象。

`EnvironmentHandle` 不暴露裸 CraftGround 对象。它内部持有该槽位的键鼠适配器，`apply(ActionTick)`
完成转译并返回 `StepOutcome`，因此设备边界不会随句柄一起交给调用方。`preview_adapter()` 克隆当前
设备状态，供调用方在不触碰环境的前提下预演转译；教师执行器正是用它在提交前拒绝非法输入。

每个实例在同一时刻只归一个 SubAgent 使用。推演请求中的 `simulate(handle, payload)` 收到的是句柄，
不是环境。

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

`capture` 会在每个 JVM 内保存同名快照。每项推演取得独占句柄后，先加载请求指定的快照（`snapshot`
为空时回到根快照），再调用请求中的 `simulate(handle, payload)`。线程池并行驱动多个独立 JVM，因此
环境仿真和 IPC 等待可以同时进行。并发上限由环境池而不是线程池决定：请求数量超过槽位时在
`EnvironmentPool` 中等待，`wait_timeout=None` 表示持续等待，给定秒数后仍无槽位则抛出
`EnvironmentPoolTimeout`。推演成功或抛出异常时租约都会归还。

快照是 JVM 本地对象。同一 ID 必须预先广播保存到所有可能接收该任务的环境，不能把一个 JVM 中的
`StructureTemplate` 直接传给另一个 JVM。

可比较推演还要求各 JVM 在保存同名快照前处于同一逻辑状态。四分支验证入口使用固定种子，冻结昼夜、
天气、生物生成和随机 tick，清除普通实体，并比对玩家位置、视角、生命、背包和视线方块。快照区域
根据实际玩家坐标计算；保存后入口主动扰动各环境，再倒档并比对状态，以验证实际恢复路径。
