# Godot 强化学习与 Minecraft BC 训练项目

本仓库保留 Godot 环境、跨进程共享内存通信、SB3 PPO 训练，以及 Minecraft VPT
行为克隆暖启动和对应的像素快塔。
当前任务是聚光灯瞄准：40 个 Godot 子环境锁步运行，Python 通过文件后端 mmap
读取图像与元数据并发送动作。

## 环境

- Python 3.11+
- Godot 4.6.1 .NET 版
- .NET 8 SDK
- Python 包：numpy、gymnasium、opencv-python、stable-baselines3、torch

Python 与 Godot 通过 mmap 通信，不使用也不需要 godot-python。

    pip install -e .
    pip install -e .[dev]

通过环境变量 GODOT_EXE 指定 Godot 可执行文件。Windows 通常需要绝对路径；
Linux 可在 PATH 中提供 godot，也可同样显式指定。

    export GODOT_EXE=/path/to/godot
    python -m train.godot_meta_rl.train_ppo --total-timesteps 100000

Minecraft BC 使用成对的 VPT `mp4 + jsonl` 数据。先查看数据参数：

    python -m train.minecraft.bc_warmstart --help

该路径只包含 VPT 数据解析、动作契约、PixelTower 和 BC 训练，不包含 Minecraft
在线环境、VPT 教师、GRPO、慢塔或旧 DINO 地图快塔。

Windows PowerShell：

    $env:GODOT_EXE = "C:\path\to\Godot_v4.6.1-stable_mono_win64.exe"
    python -m train.godot_meta_rl.train_ppo --total-timesteps 100000

Linux 需要可产生像素的 X11/Vulkan 渲染环境。Godot 的 --headless 哑渲染器
不能用于图像训练。具体显示服务和驱动安装由运行机器负责，本项目不猜测发行版包名。

## 目录

    assets/godot_meta_rl/   Godot 4.6.1 .NET 工程与环境场景
    utils/godot_rl/         mmap 协议、Godot 进程启停、PPO 工厂
    train/godot_meta_rl/    SB3 VecEnv 与 PPO 训练入口
    tests/                  共享内存协议纯单元测试

Godot 侧协议与方法说明见
[assets/godot_meta_rl/README.md](assets/godot_meta_rl/README.md) 和
[assets/godot_meta_rl/code_analysis.md](assets/godot_meta_rl/code_analysis.md)。
