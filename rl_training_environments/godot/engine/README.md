# Godot Meta RL 子系统

Godot 引擎侧的元学习强化学习采集环境。当前任务是**聚光灯瞄准**（spotlight aiming）：在随机时刻点亮房间内某个目标物体，agent 通过离散键位控制相机云台对准它。一次运行 40 个并行环境，通过共享内存与 Python 侧的训练循环 lock-step 通信。

本仓库当前只保留该 Godot 环境及其 Python 训练管线。

---

## 目录构成

本文件夹只保留 **Godot 引擎侧**资产：

| 文件 | 作用 |
|---|---|
| `TrainingCoordinator.cs` | C# 编排器：加载 40 份环境场景，发布观测并分发动作 |
| `environment_model_base.gd` | 采集环境基类：动作空间、相机云台、物理步进与通信契约 |
| `meta_environments/discrete_spotlight_environment.*` | 聚光灯瞄准任务的离散控制实现 |
| `training_main.tscn` | 训练用主场景 |
| `project.godot` / `GodotMetaReinforcementLearning.csproj` / `nuget.config` | Godot 工程与 C# 配置 |

Python 侧已按仓库分层规范迁出本文件夹：

- **Python 环境包 → `rl_training_environments/godot/`**：共享内存、进程管理、SB3 适配和 PPO 训练入口。

方法级说明见 [code_analysis.md](code_analysis.md)。

---

## 通信协议

所有 Python 脚本通过固定握手与 Godot 通信（轮询计数器 seqlock，跨平台）：

```
wait_for_observation() → read_image_observations() / read_metadata() → send_actions(...)
```

握手计数器：Godot 发布观测后 `ObsSeq+1`；Python `wait_obs` 轮询到 `ObsSeq != 已消费值` 即取帧，`send_action` 写完动作后把 `ActSeq` 置为已消费的 `ObsSeq` 作为应答；Godot 轮询 `ActSeq==ObsSeq` 才步进。收到应答前不推进，因此帧号严格 +1、无丢帧、不发动作即门控停住。

| 数据 | 规格 |
|---|---|
| 图像观测 | `(40, 128, 128, 3)` uint8 |
| 元数据 | `(40, 5)` float32：`[frameCount, steps, sim_dt, reward, done]` |
| 连续动作 | `(40, 10)` float32 |
| 离散动作 | `(40, 30)` int32 |
| Godot 场景 | `training_main.tscn` → `TrainingCoordinator.cs` |

共享内存采用**文件后端 mmap + 共享内存内轮询计数器（seqlock）**握手，Windows / Linux 自动识别，不依赖 Windows 命名内核对象。

---

## 运行

- **Windows**：直接运行，可回读到非零像素。
- **Linux 无头**：需 Xvfb + GPU 或软件 Vulkan（lavapipe）渲染才能回读到非零像素；`--headless` 哑渲染器不产出像素。

设置 `GODOT_EXE` 后运行 `train-godot-ppo`。入口启动 Godot，并由 `GodotVectorizedEnvironment` 驱动 PPO。Python 与 Godot 通过文件映射通信，不使用 `godot-python`。
