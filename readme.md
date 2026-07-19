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

旧 PixelTower BC 基线仍可使用成对的 VPT `mp4 + jsonl` 数据：

    python -m train.minecraft.behavior_cloning_warm_start --help

新的初版训练入口联合训练约 100M 时空 VLA 快塔和训练期 Dreamer-lite 潜状态
世界模型。冻结 DINOv3-S 与 MiniLM 会从 Hugging Face 下载到其标准缓存：

    python -m train.minecraft.world_model_warm_start \
        --data runs/data/vpt_stream \
        --data-manifest /path/to/vpt_manifest.jsonl \
        --stream-cache-gib 80

AutoDL 中断后使用同一配置精确续训：

    python -m train.minecraft.world_model_warm_start \
        --data runs/data/vpt_stream \
        --data-manifest /path/to/vpt_manifest.jsonl \
        --resume runs/checkpoints/minecraft_dreamer_lite/last.pt

世界模型负责动作条件化的下一视觉 latent 预测与离散 KL；奖励、继续概率、事件和
物品栏头已定义，但只有 CraftGround 回放提供对应真值后才应加入损失。训练 checkpoint
显式标记为 `minecraft_dreamer_lite_v1`，不与 PixelTower 或旧 v2 checkpoint 部分加载。

流式清单可以是本地文件或 HTTP(S) URL，每行格式如下。下载器先写 `.part`，校验
可选 SHA256 后原子发布完整的 `mp4 + jsonl` 文件对，并按 `--stream-cache-gib`
滚动淘汰旧段；私有 Hugging Face 文件可通过 `HF_TOKEN` 环境变量授权。

    {"name":"clip_0001","video_url":"https://.../clip.mp4","action_url":"https://.../clip.jsonl","video_sha256":"可选","action_sha256":"可选"}

数据目录只使用 `VPTStreamDataset` 的 clip 级 uint8 缓存，不再保留全量预载数据集、
废弃的窗口复用缓存或静默缓存参数旁路。

## 快塔 v2

`net.spatiotemporal_fast_tower.SpatiotemporalFastTower` 是不兼容旧 checkpoint 的新结构。
冻结 DINO 和冻结文本编码器在模型外运行，快塔核心不下载权重。正式默认配置约
100M 可训练参数，动作块长度 `K=4`。默认输入契约如下：

- 当前帧：`[B,576,Dv]`，对应 `18×32` 完整 patch 网格；
- 历史帧：`[B,H,576,Dv]`，内部做 `2×2` 池化后逐空间位置进行时间混合；
- 文本：完整 token 与有效位，不使用 UTF-8 字节从零学习语义；
- 控制状态：过去动作、每步 `dt`、归一化像素指点及其有效位；
- 记忆：默认 `NullMemory`，地图不是基线必需项；
- 输出：未来 `[B,K]` 上的相机两轴 11 档、互斥前后/左右/姿态/hotbar，以及五个事件按钮；部署只执行前 1–2 步后重规划。

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
    train/minecraft/                        BC 与潜动力学训练循环
    net/                    PixelTower、SpatiotemporalFastTower 与潜状态世界模型
    tests/                  共享内存协议纯单元测试

Godot 侧协议与方法说明见
[rl_training_environments/godot/engine/README.md](rl_training_environments/godot/engine/README.md) 和
[rl_training_environments/godot/engine/code_analysis.md](rl_training_environments/godot/engine/code_analysis.md)。

## 命名约定

文件、目录、模块、公开类型和普通变量使用完整且描述性的英文单词。Python 与
GDScript 文件使用 `snake_case`，C# 类型及对应文件使用 `PascalCase`。PPO、VPT、
RL、RGB、DINO 和 API 等领域标准缩写可以保留，不使用项目私有缩写。张量 Shape
说明中的 B/T/H/W/C 保留数学符号写法。
