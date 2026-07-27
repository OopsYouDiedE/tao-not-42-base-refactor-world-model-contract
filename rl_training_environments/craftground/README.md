# CraftGround 环境配置

CraftGround 基于 Minecraft Java 版，在无头 Linux 服务器上运行需要安装以下系统依赖。

## 系统依赖安装（一键命令）

在 Ubuntu/Debian 系统上，执行以下命令安装全部前置依赖：

```bash
apt-get update && apt-get install -y \
  openjdk-21-jdk \
  cmake \
  build-essential \
  xvfb \
  libx11-6 \
  libxext6 \
  libxrender1 \
  libxtst6 \
  libxi6 \
  libgl1-mesa-dev \
  libglu1-mesa-dev \
  libgl1-mesa-dri \
  libglx-mesa0 \
  libglew-dev \
  mesa-utils
```

安装完成后，设置 Java 21 为默认版本：

```bash
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
```

可将上述 `export` 写入 `~/.bashrc` 或 `/etc/environment` 以持久化。

## 依赖说明

| 包名 | 用途 |
|---|---|
| `openjdk-21-jdk` | Minecraft 运行与首次冷启动 Gradle 编译均需要 JDK 21 |
| `cmake` `build-essential` | CraftGround 首次启动时编译 C++ 原生通信模块 |
| `xvfb` | 在无头服务器上虚拟出显示器（Xvfb） |
| `libx11-6` `libxext6` `libxrender1` `libxtst6` `libxi6` | X11 窗口系统运行库 |
| `libgl1-mesa-dev` `libglu1-mesa-dev` | OpenGL 开发库（C++ 编译时的头文件） |
| `libgl1-mesa-dri` `libglx-mesa0` | Mesa OpenGL 运行时驱动 |
| `libglew-dev` | OpenGL 扩展管理库（编译时需要） |
| `mesa-utils` | OpenGL 诊断工具（`glxinfo` 等） |

## 无头服务器启动方式

在没有显示器的服务器上，必须通过 `xvfb-run` 启动所有需要渲染的脚本，
让 `MinecraftCraftGroundEnvironment` 能拿到 `DISPLAY`：

```bash
xvfb-run -a python -c "
from rl_training_environments.craftground.environment import MinecraftCraftGroundEnvironment
from craftground.screen_encoding_modes import ScreenEncodingMode

env = MinecraftCraftGroundEnvironment(seed=0, max_steps=200,
                                      screen_encoding_mode=ScreenEncodingMode.RAW)
obs = env.reset()                     # (H, W, 3) uint8 RGB
for _ in range(200):
    obs, reward, done, info = env.step(1)   # 1 = forward，见 DISCRETE_TO_V2
    if done:
        obs = env.reset()
env.close()
"
```

首次运行会触发 Gradle 冷编译 Minecraft mod 与 C++ 原生模块（见下文“首次启动编译耗时”）。

## 常见问题

### ALSA 声卡警告

无头服务器上会出现类似以下日志：

```
ALSA lib confmisc.c:855:(parse_card) cannot find card '0'
```

这是因为服务器没有音频硬件，**不影响运行**，可以安全忽略。

### 首次启动编译耗时

CraftGround 首次启动时，Gradle 会自动编译 Minecraft Mod 和 C++ 原生模块。
此过程耗时较长（约 1-5 分钟），编译结果会被缓存，后续启动不会重复编译。

### NumPy 负步长警告

CraftGround 返回的图像数组可能带有负步长，代码中已通过
`np.ascontiguousarray()` 处理，可以安全忽略相关 UserWarning。

## 设备无关控制契约的接入（`control_adapter.py`）

`control_adapter.py` 把 `control_contract` 的设备无关 `DeviceFrame` 转成本环境的 V2 22 键
动作 dict，是本目录里**唯一**知道"语义角色 → Minecraft 键名"的地方。绑定声明在
`control_contract/profiles/minecraft_mouse_keyboard.json`：20 Hz、瞄准摇杆 18°/tick、位移摇杆
8 向单档（WASD）、9 槽位直达，`map/menu/confirm/cancel/nav_*` 声明为本游戏不存在。

CraftGround 没有绝对光标通道（GUI 打开时相机增量驱动屏幕光标），因此该 profile 的
`cursor_cap_per_tick` 只有 0.1875 屏/tick——**不是**桌面鼠标那种单 tick 跳转。编译器据此把
`point` 拆成逐 tick 目标位置，适配层再把相邻 tick 的位置差换成相机增量。

标定已实测完成，常量在 `segment_text_codec.py` 顶部，不是估算值：

| 量 | 值 | 怎么测出来的 |
|---|---|---|
| 像素↔度数 | 1 px = **0.15°** | 请求 0.34°/tick 实测只走 2 px，小数被截断且不累积 |
| 每 tick 上限 | **120 px** | 继承瞄准的 18°/tick：18 / 0.15 |
| 满屏度数 | 96° 横 / 54° 纵 | 640×360 下模板匹配白箭头逐帧定位 |
| GUI 开启延迟 | **2 tick** | 按 E 后头 2 tick 的相机增量转的是世界视角 |
| 光标复位 | 每次打开回正中 | 背包不关则跨段延续 |

因为只走整数像素，规划必须按整数像素做——按度数平均摊会成比例丢量。两轴满屏度数不同，
所以适配层用 `CURSOR_DEGREES_PER_SCREEN_WIDTH` / `_HEIGHT` 两个常量，没有"统一屏幕单位"。
回归测试见 `tests/unit/test_segment_text_codec_cursor.py`。

与下面的回合录制器是**并行**的两条路：录制器用 `macro_action_compiler` 的 Turn 手写示范数据，
在线大模型控制用 `control_contract` 的 Segment。两者最终都落到同一套 V2 动作 dict。

## 大模型在线控制与数据采集

三个模块串起一条闭环：模型写文本 → 编译成逐 tick 动作 → 执行 → 实测反馈回提示词。

| 模块 | 职责 |
|---|---|
| `segment_prompt_builder.py` | 拼装提示词（8 个区，前 3 区静态走 cache） |
| `segment_text_codec.py` | 解析模型文本 + 编译成 V2 动作。面向模型宽容，面向程序严格 |
| `llm_segment_controller.py` | 跑一轮：请求模型 → 执行 → 守卫 → 记账 → 写回事实 |

跑一次采集（会真起 Minecraft，需无头显示）：

```bash
python -m rl_training_environments.craftground.run_llm_log_collection \
  --seed 1234 --model claude-sonnet-4-6 --stage collect --chain-to-stone \
  --output-directory runs/llm-log-collection/v4
```

产物是 `trajectory.json` + `frames/`。渲染成带截图的可读轨迹：

```bash
python -m rl_training_environments.craftground.render_trajectory_markdown \
  runs/llm-log-collection/v4
```

md 用相对路径引用 `frames/`，所以必须留在运行目录里。审查提示词是否夹带答案：

```bash
python -m rl_training_environments.craftground.dump_prompt > prompt_snapshot.txt
```

提示词格式说明见 `runs/llm-log-collection/prompt_format.md`。

## 手写轨迹录制器（可视化界面）

`trajectory_recorder_server.py` 是一个"模拟器式 TAS 编辑器"：用**回合（Turn）**逐条手写轨迹，
逐 tick 展开成 CraftGround V2 的 22 键动作（= 模型动作空间本身），在浏览器界面里"步进到下一个
观察点"回放、"重置到轨迹初始"，最后导出成训练用的示范数据。界面 header 显示 `recorder_version`
（当前 v3 = 回合模型 + md/json 导出），导出产物也写入 provenance。

**回合模型**：一个回合 = 大模型一次推理 = 一个观察段。回合内可**并发多个动作（组合键）**，
录入分四部分：

- **① 离散键序列**（键名直打，自由多选组合，不做互斥）：**留空时长 = 点按（`TAP_TICKS`=2 tick）**，
  **填了时长 = 定时长按**（本回合内按各自时长为真，短键提前松开，不跨回合）。含 hotbar.1-9。
- **② WASD / 姿态长按**（`forward/back/left/right/sneak/sprint`，与①隔离单算，必须带时长）：
  这些 **latch 键**的时长作为**后台倒计时预算跨回合延续**——"没设置就不覆盖"，某回合不再指定
  则沿用剩余预算继续按住，预算归零自动松开，重设同键则覆盖刷新。
- **③ 相机**（固定 `CAM_TICKS`=2 tick）：两种互斥结构——① 增量 `delta_yaw/delta_pitch`（度，
  可勾 GUI 光标语义）；② 绝对屏幕坐标 `screen_x/screen_y`（GUI 光标定位）。delta 模式单回合
  `|Δ| ≤ CAM_TICKS×CAM_MAX_DEG`（=36°），超限**后端报错并回传界面**（否则 BC 编码端
  `action_contract.deg_to_bins` 会静默截断，导致录制执行值与可学习目标不一致）。
- **④ 纯等待** `wait_ticks`：全空回合用它推进时间轴（等熔炉烧/等岩浆流），期间 latch 键继续。

离散键**不做互斥校验**（`forward+back` 等允许）；跨互斥组的同按会在策略解码端
`net/action_token_codec.py` 被消解，录制器只在界面提示、不阻止。

**观察点语义**：观察点（observation point）= 一次推理边界 = 一条训练样本的切点。相邻观察点
之间是模型"盲执行"的一段动作（段内不看帧、不推理）。落点 = 每个推进回合的结束边界必有 +
尾段（消费剩余 latch 预算）每 2 tick 一个 + 任何超过 `--max-blind-ticks` 的连续盲段自动补插。

**序列内 mc 命令 vs 即时命令**（界面严格区分）：
- **序列内 mc 宏**（`kind=="mc"`，标注"可复现·进导出"）：进轨迹，导出后随回合序列确定性重现，
  世界搭建（`setblock` 工作台/熔炉）用它。
- **即时命令**（右侧常驻框，走 `env.add_command`，标注"仅预览·不进导出·不可复现"）：只作用于
  活 env 当前预览、不进导出；发送会推进活 env 若干 tick 并清空已存观察帧（步进前搭世界/调试用）。

启动（无头服务器需 xvfb；默认 `normal` 真实地形 + `survival`）：

```bash
xvfb-run -a python -m rl_training_environments.craftground.trajectory_recorder_server \
  --http-port 8897 --world normal --gamemode survival \
  --extra-command "give @p oak_planks 64"
```

浏览器打开 `http://127.0.0.1:8897/`。左侧显示当前帧 + 观察点时间线（点空白处手插观察点，
点橙点重放跳转），右侧组回合并管理序列。按钮：**步进**（执行到下一观察点并存该帧）、
**死亡重置**（`fast_reset` 回初始态，只重置玩家不重置世界方块，亚秒）、**完全重置**（冷重开
env ~30s，回干净世界/存档）、**导出**（从干净初始态完整重放，落 `runs/craftground-trajectories/<name>/`：
`trajectory.md` 人读回合表 + `trajectory.json` 机读回合 spec（**不含逐 tick 展开**，下游读 json
后自行跑 `compile_macros` 重算）+ 逐观察点帧 png + `frames.mp4`）。

**约束与实测事实**（见根目录记忆 `craftground-recorder-capability-probe-2026-07-25`、
`craftground-reset-and-save-facts-2026-07-25`）：CraftGround 无真 savestate——`fast_reset`（亚秒级）
靠 `/kill @p`+重生实现，只重置玩家不重置世界方块（`SaveWorldMixin` 禁用 `saveAll`，会话内改动
永不落盘，存档天然只读）；`fast_reset=False` 冷重开=完全重置，世界回 config/存档定义的干净初始态。
导出固定走冷重开保证 survival 挖改地形后仍能从干净世界确定性重现。录制器持有**一个常驻 env**，
30 秒冷启动整个会话只付一次。分辨率固定 640×360，与 solaris 渲染器口径一致
（`observation_spaces.OBS_SHAPE_NATIVE`）。
