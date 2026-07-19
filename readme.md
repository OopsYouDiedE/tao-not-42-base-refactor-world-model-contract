# Godot 强化学习与 Minecraft BC 训练项目

本仓库保留 Godot 环境、CraftGround 在线环境、Minecraft VPT 行为克隆暖启动，
以及 v1/v2 两代像素快塔。
当前任务是聚光灯瞄准：40 个 Godot 子环境锁步运行，Python 通过文件后端 mmap
读取图像与元数据并发送动作。

## 环境

- Python 3.11+
- Godot 4.6.1 .NET 版
- .NET 8 SDK
- CraftGround 运行所需的 Java 21
- Python 包：craftground、numpy、gymnasium、opencv-python、stable-baselines3、torch

Python 与 Godot 通过 mmap 通信，不使用也不需要 godot-python。
CraftGround 使用实际的 `craftground` Python 包；本项目不引入 MineRL 或 VPT 教师依赖。

    pip install -e .
    pip install -e .[dev]

通过环境变量 GODOT_EXE 指定 Godot 可执行文件。Windows 通常需要绝对路径；
Linux 可在 PATH 中提供 godot，也可同样显式指定。

    export GODOT_EXE=/path/to/godot
    python -m rl_training_environments.godot.train_ppo --total-timesteps 100000

Minecraft BC 使用成对的 VPT `mp4 + jsonl` 数据。先查看数据参数：

    python -m train.minecraft.behavior_cloning_warm_start --help

该路径只包含 VPT 数据解析、动作契约、PixelTower 和 BC 训练，不包含 Minecraft
在线环境、VPT 教师、GRPO、慢塔或旧 DINO 地图快塔。

## 快塔 v2

`net.spatiotemporal_fast_tower.SpatiotemporalFastTower` 是不兼容旧 checkpoint 的新结构。冻结 DINO 和冻结
文本编码器在模型外运行并缓存 token，快塔核心不下载权重。默认输入契约如下：

- 当前帧：`[B,576,Dv]`，对应 `18×32` 完整 patch 网格；
- 历史帧：`[B,H,576,Dv]`，内部做 `2×2` 池化后逐空间位置进行时间混合；
- 文本：完整 token 与有效位，不使用 UTF-8 字节从零学习语义；
- 控制状态：过去动作、每步 `dt`、归一化像素指点及其有效位；
- 记忆：默认 `NullMemory`，地图不是基线必需项；
- 输出：相机两轴 11 档、互斥前后/左右/姿态/hotbar，以及五个事件按钮。

结构化动作可以展开为 CraftGround 的 20 键概率视图，但采样时应从互斥类别分布
采样，不能独立采出前后同按、左右同按或多个 hotbar。

Windows PowerShell：

    $env:GODOT_EXE = "C:\path\to\Godot_v4.6.1-stable_mono_win64.exe"
    python -m rl_training_environments.godot.train_ppo --total-timesteps 100000

Linux 需要可产生像素的 X11/Vulkan 渲染环境。Godot 的 --headless 哑渲染器
不能用于图像训练。具体显示服务和驱动安装由运行机器负责，本项目不猜测发行版包名。

## 目录

    rl_training_environments/godot/         Godot 通信、环境适配与训练入口
    rl_training_environments/godot/engine/  Godot 4.6.1 .NET 工程与场景
    rl_training_environments/craftground/  CraftGround 环境、奖励、回放与世界快照
    datasets/vpt/                           VPT 视频数据与原始动作契约
    train/minecraft/                        BC 训练循环
    net/                    PixelTower 与 SpatiotemporalFastTower
    tests/                  共享内存协议纯单元测试

Godot 侧协议与方法说明见
[rl_training_environments/godot/engine/README.md](rl_training_environments/godot/engine/README.md) 和
[rl_training_environments/godot/engine/code_analysis.md](rl_training_environments/godot/engine/code_analysis.md)。

## 命名约定

文件、目录、模块、公开类型和普通变量使用完整且描述性的英文单词。Python 与
GDScript 文件使用 `snake_case`，C# 类型及对应文件使用 `PascalCase`。PPO、VPT、
RL、RGB、DINO 和 API 等领域标准缩写可以保留，不使用项目私有缩写。张量 Shape
说明中的 B/T/H/W/C 保留数学符号写法。
