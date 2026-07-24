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
