# Godot Meta RL 项目代码结构与方法调用关系文档

本文档梳理 `godot_meta_rl` 的核心作用、方法说明与调用关系。

> **目录现状**：本文件夹现仅保留 **Godot 引擎侧**资产（`Main.cs`、`model_base.gd`、`mata_envs/env_spotlight_discrete.*`、`train_main.tscn`、`project.godot`、`元学习任务.csproj`、`nuget.config`）。Python 侧编排已按仓库分层规范迁出：
> - **工厂/可复用基础设施 → `utils/godot_rl/`**：`shared_mem_env.py`（跨平台共享内存驱动 `GodotTrainEnv` + 布局常量 + `shm_path`/`warmup`）、`launch.py`（`launch_godot`/`kill_godot`）、`ppo_factory.py`（`build_model`/`make_buffer`/`extract_small`/`bind_small`）。
> - **独特不可复用的对接桥 → `train/godot_meta_rl/`**：`vec_env.py`（`GodotVecEnv`/`RolloutProgress`，SB3 VecEnv 适配，当前该目录唯一 Python 文件）。
> - **已在重构清理中删除**（退役 PPO 管线 + 诊断/协议测试，见 git 历史）：`train_ppo.py`/`train_ppo_async.py`/`train_ppo_2proc.py`、`smoke.py`、`diag_montage.py`、`async_min.py`、`cleanup_workspace.py`、`test_shm_integration.py`/`test_step_modes.py`/`test_frame_completeness.py`/`test_reader_compare.py`。下文第 1 节方法表的「调用方」列仍列有这些已删脚本，保留作迁移参考；自动化测试一节(原第 4 节)已整体移除。
> - 下表中的 Python 模块名（`rl_train_env`/`train_ppo`/…）为迁移前旧名，方法说明仍然有效，对应新位置见上。
>
> **跨平台**：共享内存改为【文件后端 mmap】+【共享内存内轮询计数器(seqlock)】握手，Win/Linux 自动识别，不再依赖 Windows 命名内核对象。Linux 无头运行需 Xvfb + GPU/软件 Vulkan(lavapipe) 渲染才能回读到非零像素（`--headless` 哑渲染器不出像素）。
>
> **历史**：旧协议文件（`shm_reader.py`、`shm_reader_fast.py`、`SharedMemCommunicator.cs`、`rl_agent_loop.py`、`main.tscn`）已全部删除。

---

## 目录
1. [Python 对接桥与共享内存驱动](#1-python-对接桥与共享内存驱动)
   - [shared_mem_env.py（GodotTrainEnv 驱动）](#shared_mem_envpy)
   - [vec_env.py（GodotVecEnv / RolloutProgress）](#vec_envpy)
2. [Godot C# 编排模块](#2-godot-c-编排模块)
   - [Main.cs](#maincs)
3. [Godot GDScript 环境与任务逻辑模块](#3-godot-gdscript-环境与任务逻辑模块)
   - [model_base.gd](#model_basegd)
   - [env_spotlight_discrete.gd](#env_spotlight_discretegd)

---

## 统一协议说明

所有 Python 脚本通过以下固定握手与 Godot 通信（轮询计数器 seqlock，跨平台）：

```
wait_obs()  ->  read_images() / read_meta()  ->  send_action(cont, disc)
```

握手计数器：Godot 发布观测后 `ObsSeq+1`；Python `wait_obs` 轮询到 `ObsSeq != 已消费值` 即取帧，`send_action`
写完动作把 `ActSeq` 置为已消费的 `ObsSeq` 作为应答；Godot 轮询 `ActSeq==ObsSeq` 才步进（收到应答前绝不推进 →
帧号严格 +1、无丢帧、不发动作即门控停住）。

| 参数 | 规格 |
|---|---|
| 图像观测 | `(40, 128, 128, 3)` uint8 |
| 元数据 | `(40, 5)` float32: `[frameCount, steps, sim_dt, reward, done]` |
| 连续动作 | `(40, 10)` float32 |
| 离散动作 | `(40, 30)` int32 |
| Godot 场景 | `train_main.tscn` -> `Main.cs` |

---

## 1. Python 对接桥与共享内存驱动

> 下方方法表的「调用方」列保留了清理前的历史调用脚本（`train_ppo*.py`/`diag_montage.py`/`test_*.py` 等均已删除），仅 `GodotTrainEnv`(在 `utils/godot_rl/shared_mem_env.py`)与 `GodotVecEnv`(在 `vec_env.py`)契约仍然有效。

### `shared_mem_env.py`
* **作用**：Python 端底层驱动 `GodotTrainEnv`（原 `rl_train_env.py`，现迁至 `utils/godot_rl/`）。封装 40 环境的共享内存布局（图像区 + 5字段元数据 + 10连续/30离散动作），通过【文件后端 mmap + 共享内存内轮询计数器(seqlock)】跨平台与 `Main.cs` lock-step 握手；计时器精度辅助函数供基准测试使用。
* **方法标注与调用关系**：

| 方法/函数名 | 方法说明 | 调用该方法的函数/组件 |
| :--- | :--- | :--- |
| `set_timer_resolution(ms=1)` | 提高 Windows 计时器精度，使 `time.sleep` 在毫秒级更可靠。从 `rl_agent_loop.py`（已删除）合并而来。 | <li>`test_frame_completeness.py` -> `main()`</li><li>`test_reader_compare.py` -> `main()`</li> |
| `reset_timer_resolution(ms=1)` | 还原 Windows 计时器精度。 | <li>`test_frame_completeness.py` -> `main()`</li><li>`test_reader_compare.py` -> `main()`</li> |
| `GodotTrainEnv.__init__(self, connect_timeout_s=40.0)` | 初始化 `GodotTrainEnv` 实例，打开用于训练同步的事件与共享内存。 | <li>`train_ppo.py` -> `GodotVecEnv.__init__()`</li><li>`train_driver.py` -> `main()`</li><li>`diag_montage.py` -> `main()`</li><li>`test_shm_integration.py` -> `main()`</li><li>`test_frame_completeness.py` -> `main()`</li><li>`test_reader_compare.py` -> `main()`</li><li>`test_step_modes.py` -> `run_mode()`</li> |
| `GodotTrainEnv._open(factory, deadline)` | 静态辅助方法，在截止时间前循环重试打开指定内核对象（事件或共享内存）。 | <li>`GodotTrainEnv.__init__()` 内部调用</li> |
| `GodotTrainEnv.wait_obs(self, timeout_ms=2000)` | 等待 Godot 端的观测就绪事件 `obs_ready`。 | <li>`train_ppo.py` -> `GodotVecEnv.reset()`</li><li>`train_ppo.py` -> `GodotVecEnv.step_wait()`</li><li>`train_driver.py` -> `main()`</li><li>`diag_montage.py` -> `main()`</li><li>`test_shm_integration.py` -> `main()`</li><li>`test_frame_completeness.py` -> `run_phase()`、`main()` (预热)</li><li>`test_reader_compare.py` -> `run_load()`、`main()` (预热)</li><li>`test_step_modes.py` -> `run_mode()`</li> |
| `GodotTrainEnv.read_meta(self)` | 从共享内存中读取 40 个环境的 5 字段元数据：`[frameCount, steps, sim_dt, reward, done]`。 | <li>`train_ppo.py` -> `GodotVecEnv._read_obs()`</li><li>`train_driver.py` -> `main()`</li><li>`diag_montage.py` -> `main()`</li><li>`test_shm_integration.py` -> `main()`</li><li>`test_frame_completeness.py` -> `run_phase()`</li><li>`test_reader_compare.py` -> `run_load()`</li><li>`test_step_modes.py` -> `run_mode()`</li> |
| `GodotTrainEnv.read_images(self)` | 从共享内存首部读取 40 个环境的 `128x128x3` uint8 图像。 | <li>`train_ppo.py` -> `GodotVecEnv._read_obs()`</li><li>`train_driver.py` -> `main()`</li><li>`diag_montage.py` -> `main()`</li><li>`test_shm_integration.py` -> `main()`</li><li>`test_frame_completeness.py` -> `run_phase()`</li><li>`test_reader_compare.py` -> `run_load()`</li> |
| `GodotTrainEnv.send_action(self, cont, disc)` | 写入连续动作数组 `cont:(N,10) float32` 与离散动作数组 `disc:(N,30) int32` 到共享内存，然后置位 `act_ready`。 | <li>`train_ppo.py` -> `GodotVecEnv.step_wait()`</li><li>`train_driver.py` -> `main()`</li><li>`diag_montage.py` -> `main()`</li><li>`test_shm_integration.py` -> `main()`</li><li>`test_frame_completeness.py` -> `run_phase()`、`main()` (预热)</li><li>`test_reader_compare.py` -> `run_load()`、`main()` (预热)</li><li>`test_step_modes.py` -> `run_mode()`</li> |
| `GodotTrainEnv.close(self)` | 关闭共享内存连接。 | <li>`train_ppo.py` -> `GodotVecEnv.close()`</li><li>`train_driver.py` -> `main()`</li><li>`diag_montage.py` -> `main()`</li><li>`test_shm_integration.py` -> `main()`</li><li>`test_frame_completeness.py` -> `main()`</li><li>`test_reader_compare.py` -> `main()`</li><li>`test_step_modes.py` -> `run_mode()`</li> |

---

### `vec_env.py`
* **作用**：把 40 并行环境的 lock-step 机制封装成 SB3 兼容的 `VecEnv`（`GodotVecEnv`）并提供进度回调 `RolloutProgress`，供训练入口调用。下表中以 `main()` 为代表的训练装配行原属已删除的 `train_ppo.py`，保留作接口契约参考。
* **方法标注与调用关系**：

| 方法/函数名 | 方法说明 | 调用该方法的函数/组件 |
| :--- | :--- | :--- |
| `RolloutProgress.__init__(self)` | 进度回调类构造函数，初始化计时器。 | <li>`main()` -> `RolloutProgress()`</li> |
| `RolloutProgress._on_training_start(self)` | 训练开始时回调，记录初始时间 `_t0`。 | <li>SB3 内部训练生命周期调用</li> |
| `RolloutProgress._on_step(self)` | 每步训练回调，直接返回 `True` 允许继续训练。 | <li>SB3 内部训练生命周期调用</li> |
| `RolloutProgress._on_rollout_end(self)` | 在每次 Rollout 结束时，计算最近的回合奖励均值 `ep_rew_mean`、回合长度均值并输出整行紧凑进度。 | <li>SB3 内部训练生命周期调用</li> |
| `GodotVecEnv.__init__(self, connect_timeout_s=60.0)` | 构建 Gym 观测空间与动作空间，并初始化底层的 `GodotTrainEnv` 驱动。 | <li>`main()` -> `GodotVecEnv()`</li> |
| `GodotVecEnv.reset(self)` | 等待首帧观测并调用 `_read_obs` 返回初始状态。 | <li>SB3 训练引擎 (如 PPO.learn) 初始调用</li> |
| `GodotVecEnv.step_async(self, actions)` | 异步接收模型传入的动作包，将动作临时缓存到内部变量。 | <li>SB3 训练引擎主循环第一阶段</li> |
| `GodotVecEnv.step_wait(self)` | 将动作数据按 10连续/30离散 格式写入共享内存并通知 Godot 推进物理，然后阻塞等待 Godot 渲染并产出新观测，取出奖励、done 标志和必要的信息返回。 | <li>SB3 训练引擎主循环第二阶段</li> |
| `GodotVecEnv._read_obs(self)` | 从共享内存提取当前各环境的图像以及 `sim_dt` 并组装为 Gymnasium 字典观测空间。 | <li>`GodotVecEnv.reset()`</li><li>`GodotVecEnv.step_wait()`</li> |
| `GodotVecEnv.close(self)` | 关闭下属的 `GodotTrainEnv` 通道。 | <li>`main()` 退出清理</li> |
| `GodotVecEnv._resolve(self, indices)` | 辅助方法：将环境索引值解析为数组形式。 | <li>`GodotVecEnv.get_attr()`</li><li>`GodotVecEnv.env_method()`</li><li>`GodotVecEnv.env_is_wrapped()`</li> |
| `GodotVecEnv.get_attr(...)` | SB3 `VecEnv` 抽象接口：获取环境的指定属性。 | <li>SB3 或用户调试代码</li> |
| `GodotVecEnv.set_attr(...)` | SB3 `VecEnv` 抽象接口：设置环境的指定属性。 | <li>SB3 内部管理器</li> |
| `GodotVecEnv.env_method(...)` | SB3 `VecEnv` 抽象接口：调用子环境的方法。 | <li>SB3 内部管理器</li> |
| `GodotVecEnv.env_is_wrapped(...)` | SB3 `VecEnv` 抽象接口：判断环境是否套有指定的 Wrapper。 | <li>SB3 内部管理器</li> |
| `main()` | 训练主函数：通过 Subprocess 开启 Godot，拼装 SB3 `VecMonitor` 及 `VecFrameStack`，构造并启动 PPO 模型进行训练，训练完后安全关闭 Godot 并保存模型。 | <li>Python 脚本入口 `if __name__ == "__main__"` 执行</li> |

---

## 2. Godot C# 编排模块

### `Main.cs`
* **作用**：同步 (lock-step) 训练的多环境编排器（`train_main.tscn` 的主脚本）。加载 40 份 `env_spotlight_discrete.tscn`，使用事件握手发布所有环境的观测（图像 + 5字段元数据），并分发 Python 端写回的 10连续/30离散 动作包。
* **方法标注与调用关系**：

| 方法/函数名 | 方法说明 | 调用该方法的函数/组件 |
| :--- | :--- | :--- |
| `EnvGet(string k)` | 静态工具方法：获取系统环境变量，用于在启动时覆盖步进参数。 | <li>`Main._Ready()`</li> |
| `_Ready()` | Godot 生命周期就绪入口。读取并覆盖外部运行配置（步进模式、物理频率），克隆 40 个环境场景并作为子节点挂载，按 `RL_SHM_PATH`/临时目录创建**文件后端**共享内存并清零握手计数器（ObsSeq/ActSeq）。 | <li>Godot 引擎在节点加载时自动调用</li> |
| `_Process(double delta)` | Godot 帧更新入口。如果当前不是正在等待 Action 的状态，则先调用 `PublishObservation()`；接着等待并读取 Python 侧写入的 `ActReady` 动作包，按固定或随机步进数分别调用各环境的 `set_action` 和 `step_render`。 | <li>Godot 引擎在每一渲染帧时自动调用</li> |
| `PublishObservation()` | 图像回读与元数据组装。主线程循环抓取 40 个 SubViewport 的图像和 reward/done 数据，多线程并行将格式转换为 RGB8 并写入共享内存。若有环境 done，则重置之。最后触发 `_obsReady`。 | <li>`Main._Process()`</li> |
| `_ExitTree()` | 退出生命周期，释放共享内存映射与后备文件句柄，防止内存泄漏。 | <li>Godot 引擎在节点销毁时自动调用</li> |

---

## 3. Godot GDScript 环境与任务逻辑模块

### `model_base.gd`
* **作用**：所有元学习强化学习采集环境的基类。封装了公共的动作空间（10连续+30离散）、相机云台控制、运动阻尼与夹紧逻辑、时间计算及统一的 Python-C# 契约交互接口。
* **方法标注与调用关系**：

| 方法/函数名 | 方法说明 | 调用该方法的函数/组件 |
| :--- | :--- | :--- |
| `_ready() -> void` | 节点就绪。初始化动作缓冲，判定是否处于 standalone 模式，调用子类 `_setup()`，设置 Viewport 分辨率并依据模式启用/禁用实时物理循环 `_physics_process`。 | <li>Godot 引擎在加载节点时自动调用</li> |
| `set_action(cont: PackedFloat32Array, disc: PackedInt32Array) -> void` | 动作接收接口。由 C# 端 `Main` 显式调用，将动作值压入基础类的底层缓冲。 | <li>`Main.cs` -> `_Process()`</li> |
| `physics_step(dt: float) -> void` | 执行单个物理步进：调用 `_angular_accel()` 获得角加速度，积分计算 pitch/yaw 轴速度并施加阻尼，更新相机旋转，限制 pitch 在 `[-60°, 60°]`。随后累加仿真时间并调用子类的 `_task_physics(dt)` 任务逻辑。 | <li>`model_base.gd` -> `step_render()`</li><li>`model_base.gd` -> `_physics_process()` (独立运行)</li> |
| `step_render(n_steps: int, dt: float) -> void` | 每帧渲染接口。由 C# `Main` 显式调用，执行 `n_steps` 次 `physics_step` 物理迭代，更新渲染间隔时长并调用 `_compute_reward()` 计算并缓存当前奖励。 | <li>`Main.cs` -> `_Process()`</li> |
| `get_obs_image() -> Image` | 获取当前 SubViewport 渲染的图像内容。 | *(目前由 C# 端直接获取 Viewport Texture 绕过，保留作兼容备用)* |
| `get_info() -> Dictionary` | 构造并返回本次物理推进的详细结果数据包（reward、done、sim_dt、steps）。 | *(保留接口)* |
| `get_reward() -> float` | 获取本次渲染对应的缓存奖励数值。 | *(保留接口)* |
| `get_done() -> bool` | 获取当前环境是否命中或结束的回合终结状态。 | *(保留接口)* |
| `get_sim_dt() -> float` | 获取最近一次推进代表的仿真时长。 | <li>`env_spotlight_discrete.gd` -> `_on_standalone_tick()`</li> |
| `get_reward_done() -> Vector2` | C# 专属合一接口：以 Vector2 一次性返回缓存的 reward 和 done，减少跨语言调用次数。 | <li>`Main.cs` -> `PublishObservation()`</li> |
| `reset() -> void` | 重置回合。清理偏航/俯仰角度、角速度、仿真时间与奖励状态，重设相机，并调子类的 `_reset_task()`。 | <li>`model_base.gd` -> `_ready()`</li><li>`Main.cs` -> `PublishObservation()` (Done 自动重置)</li><li>`env_spotlight_discrete.gd` -> `_on_standalone_tick()` (独立测试重置)</li> |
| `_setup() -> void` *(虚函数)* | 子类建场景及初始化的重写入口。 | <li>`model_base.gd` -> `_ready()`</li> |
| `_angular_accel() -> Vector2` *(虚函数)* | 子类依据动作计算角加速度的重写入口。 | <li>`model_base.gd` -> `physics_step()`</li> |
| `_task_physics(_dt: float) -> void` *(虚函数)* | 子类各物理步迭代的任务侧更新重写入口。 | <li>`model_base.gd` -> `physics_step()`</li> |
| `_reset_task() -> void` *(虚函数)* | 子类重置任务侧局部状态的重写入口。 | <li>`model_base.gd` -> `reset()`</li> |
| `_compute_reward() -> float` *(虚函数)* | 子类计算当前渲染帧奖励的重写入口。 | <li>`model_base.gd` -> `step_render()`</li><li>`model_base.gd` -> `_physics_process()`</li> |
| `_is_done() -> bool` *(虚函数)* | 子类计算回合是否终结的重写入口。 | <li>`model_base.gd` -> `get_info()`</li><li>`model_base.gd` -> `get_done()`</li><li>`model_base.gd` -> `get_reward_done()`</li> |
| `_standalone_input() -> void` *(虚函数)* | 独立自运行测试模式下，读取键盘输入的重写入口。 | <li>`model_base.gd` -> `_physics_process()`</li> |
| `_on_standalone_tick() -> void` *(虚函数)* | 独立自运行测试模式下，帧刷新的 HUD 及状态检查重写入口。 | <li>`model_base.gd` -> `_physics_process()`</li> |
| `_physics_process(delta: float) -> void` | 仅在独立运行模式下启用的实时驱动函数，以真实 delta 模拟推进并执行自驱动测试。 | <li>Godot 引擎在物理帧更新时自动调用 (仅 standalone)</li> |

---

### `env_spotlight_discrete.gd`
* **作用**：聚光灯瞄准任务的离散控制实现，继承自 `ModelBase`。实现了随机生成目标物品、随机时刻触发聚光灯照亮特定物体、计算相机与物体的夹角误差、执行离散键位运动控制以及两阶段差分加稠密惩罚奖励机制。
* **方法标注与调用关系**：

| 方法/函数名 | 方法说明 | 调用该方法的函数/组件 |
| :--- | :--- | :--- |
| `_setup() -> void` | 初始化任务。混入 Instance ID 产生唯一随机种子，指定相机与视口，构建任务场景。 | <li>`model_base.gd` -> `_ready()`</li> |
| `_build_world() -> void` | 降低默认环境光以凸显聚光灯，创建 SpotLight3D 节点，并在房间内随机大小、材质和色彩实例化 `N_TARGETS` 个候选物体。 | <li>`env_spotlight_discrete.gd` -> `_setup()`</li> |
| `_build_room() -> void` | 拼装包含天花板、地板及红绿蓝黄不同颜色墙壁的六面房间盒子，给墙壁赋予棋盘格纹理以提供旋转方向的视觉参考。 | <li>`env_spotlight_discrete.gd` -> `_build_world()`</li> |
| `_make_checker_texture() -> ImageTexture` | 算法生成一张 32x32 灰白棋盘格纹理图。 | <li>`env_spotlight_discrete.gd` -> `_build_room()`</li> |
| `_dir_from(yaw: float, pitch: float) -> Vector3` | 将球面角坐标转为三维直角方向向量。 | <li>`env_spotlight_discrete.gd` -> `_reposition_targets()`</li> |
| `_reposition_targets() -> void` | 重随机各候选物体在安全仰角下的三维空间位置。 | <li>`env_spotlight_discrete.gd` -> `_reset_task()`</li> |
| `_reset_task() -> void` | 重置任务：将聚光灯置为不可见，随机生成照亮发生的延迟时刻 `_trigger_time`，并重排场景内所有候选物体的位置。 | <li>`model_base.gd` -> `reset()`</li> |
| `_light_target() -> void` | 随机抽选某一目标物体，将其表面用 SpotLight3D 强光打亮，开始记录其初始角误差和重设差分跟踪。 | <li>`env_spotlight_discrete.gd` -> `_task_physics()`</li> |
| `_task_physics(dt: float) -> void` | 物理更新钩子：检测在未打亮前是否已达触发时刻；若已亮起，则累加亮起暴露时长。 | <li>`model_base.gd` -> `physics_step()`</li> |
| `_aim_error() -> float` | 计算相机当前中央朝向向量与物体方向向量的夹角弧度差（误差）。 | <li>`env_spotlight_discrete.gd` -> `_light_target()`</li><li>`env_spotlight_discrete.gd` -> `_compute_reward()`</li><li>`env_spotlight_discrete.gd` -> `_is_done()`</li><li>`env_spotlight_discrete.gd` -> `_on_standalone_tick()`</li> |
| `_angular_accel() -> Vector2` | 读入 `_disc` 数组前 4 个元素（上/下/左/右），换算得到俯仰和偏航两个轴向的运动角加速度。 | <li>`model_base.gd` -> `physics_step()`</li> |
| `_compute_reward() -> float` | 稠密差分奖励逻辑：未点亮时奖励为0；点亮后，提供"上一帧夹角误差 - 当前夹角误差"的差分项，额外减去基于当前误差大小及帧时间的积分扣分惩罚项。 | <li>`model_base.gd` -> `step_render()`</li><li>`model_base.gd` -> `_physics_process()`</li> |
| `_is_done() -> bool` | 判定回合是否中止。条件：当前夹角误差已小于 `deg_to_rad(hit_threshold_deg)`（瞄准成功），或灯泡亮起时长已超 `MAX_AIM_SIM` 秒（超时）。 | <li>`model_base.gd` -> `get_info()`</li><li>`model_base.gd` -> `get_done()`</li><li>`model_base.gd` -> `get_reward_done()`</li><li>`env_spotlight_discrete.gd` -> `_on_standalone_tick()`</li> |
| `_standalone_input() -> void` | 独立运行模式键盘动作采集。读取方向键，将其压入对应的 `_disc` 方向变量中。 | <li>`model_base.gd` -> `_physics_process()`</li> |
| `_on_standalone_tick() -> void` | Standalone UI 刷新。累加测试奖励，并在屏幕上输出多行 HUD 信息；检测到 Done 时主动执行 reset 以开启下一轮测试。 | <li>`model_base.gd` -> `_physics_process()`</li> |

---

> 自动化测试模块（`test_shm_integration.py`/`test_step_modes.py`/`test_frame_completeness.py`/`test_reader_compare.py`：共享内存协议握手 / 步进模式 / 帧完整性 / 吞吐基准）已在重构清理中整体删除；如需回归请基于 `utils/godot_rl/shared_mem_env.py` 的 `GodotTrainEnv` 契约重建。
