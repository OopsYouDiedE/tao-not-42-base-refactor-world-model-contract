# Godot Meta RL 子系统

Godot 引擎侧的元学习强化学习采集环境。当前任务是**聚光灯瞄准**（spotlight aiming）：在随机时刻点亮房间内某个目标物体，agent 通过离散键位控制相机云台对准它。一次运行 40 个并行环境，通过共享内存与 Python 侧的训练循环 lock-step 通信。

本子系统与主项目（世界模型基座）相对独立，只因服务于同一目标而放在同一仓库联合维护。

---

## 目录构成

本文件夹只保留 **Godot 引擎侧**资产：

| 文件 | 作用 |
|---|---|
| `Main.cs` | C# 编排器：加载 40 份环境场景，发布观测、分发动作（`train_main.tscn` 主脚本） |
| `model_base.gd` | 所有采集环境的基类：动作空间、相机云台、物理步进、Python-C# 契约 |
| `mata_envs/env_spotlight_discrete.*` | 聚光灯瞄准任务的离散控制实现 |
| `train_main.tscn` | 训练用主场景 |
| `project.godot` / `元学习任务.csproj` / `nuget.config` | Godot 工程与 C# 依赖配置 |

Python 侧已按仓库分层规范迁出本文件夹：

- **可复用基础设施 → `utils/godot_rl/`**：`shared_mem_env.py`（共享内存驱动 `GodotTrainEnv` + 布局常量）、`launch.py`（`launch_godot`/`kill_godot`）、`ppo_factory.py`（`build_model`/`make_buffer` 等）。
- **不可复用的对接桥 → `train/godot_meta_rl/`**：`vec_env.py`（`GodotVecEnv` SB3 适配 + `RolloutProgress`）。

方法级说明见 [code_analysis.md](code_analysis.md)。

---

## 通信协议

所有 Python 脚本通过固定握手与 Godot 通信（轮询计数器 seqlock，跨平台）：

```
wait_obs()  ->  read_images() / read_meta()  ->  send_action(cont, disc)
```

握手计数器：Godot 发布观测后 `ObsSeq+1`；Python `wait_obs` 轮询到 `ObsSeq != 已消费值` 即取帧，`send_action` 写完动作后把 `ActSeq` 置为已消费的 `ObsSeq` 作为应答；Godot 轮询 `ActSeq==ObsSeq` 才步进。收到应答前不推进，因此帧号严格 +1、无丢帧、不发动作即门控停住。

| 数据 | 规格 |
|---|---|
| 图像观测 | `(40, 128, 128, 3)` uint8 |
| 元数据 | `(40, 5)` float32：`[frameCount, steps, sim_dt, reward, done]` |
| 连续动作 | `(40, 10)` float32 |
| 离散动作 | `(40, 30)` int32 |
| Godot 场景 | `train_main.tscn` → `Main.cs` |

共享内存采用**文件后端 mmap + 共享内存内轮询计数器（seqlock）**握手，Windows / Linux 自动识别，不依赖 Windows 命名内核对象。

---

## 运行

- **Windows**：直接运行，可回读到非零像素。
- **Linux 无头**：需 Xvfb + GPU 或软件 Vulkan（lavapipe）渲染才能回读到非零像素；`--headless` 哑渲染器不产出像素。

训练入口在 Python 侧（`train/godot_meta_rl/` + `utils/godot_rl/`）：由 `launch_godot` 启动 Godot 进程，`GodotVecEnv` 作为 SB3 `VecEnv` 驱动 PPO 训练。
