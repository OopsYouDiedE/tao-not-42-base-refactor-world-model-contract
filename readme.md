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
- Python 包：craftground、numpy、gymnasium、opencv-python-headless、stable-baselines3、torch

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

新的训练入口联合训练约 97M 时空 VLA 快塔和约 114M 训练期
Dreamer-lite 潜状态世界模型。冻结 DINOv3-S 与 MiniLM 会从 Hugging Face 下载到
其标准缓存。主课程不一次下载五组 1.55 TB 数据，而采用三个阶段：

1. `foundation / 7xx`：通用早期生存、移动、采集和合成；
2. `construction / 9xx`：方块放置、hotbar 与局部空间控制；
3. `long_horizon / 10xx`：钻石镐任务的长程采集和技术树顺序。

## AutoDL 全课程训练

选择已经正确安装 CUDA PyTorch、且 BF16 可用的 GPU 镜像。课程入口会先实际执行
一次 CUDA BF16 矩阵乘法探针，失败时会在下载大数据前终止；不需要安装 Mamba、
FlashAttention 或 MineStudio 本体。训练使用 PyTorch 原生 attention、BF16 autocast、
TF32 和 fused AdamW，不把尚无稳定训练链的 NVFP4 当作默认精度。

在仓库根目录安装并先查看远端课程：

    python -m pip install -e .
    python -m train.minecraft.autodl_curriculum --dry-run

把数据、Hugging Face 缓存和 checkpoint 放到 AutoDL 数据盘，然后直接启动：

    export TAO_STORAGE_ROOT=/root/autodl-tmp/tao-training
    export HF_HOME=$TAO_STORAGE_ROOT/huggingface
    python -m train.minecraft.autodl_curriculum \
        --data-root $TAO_STORAGE_ROOT/minestudio \
        --output $TAO_STORAGE_ROOT/checkpoints

如果准备先释放 GPU，可以在不带 CUDA 的实例上只下载数据。完整 action/meta_info 与
图像分片可以分别执行；重复执行相同命令时，Hugging Face 会按本地元数据和远端 ETag
复用已经完整下载且未变化的文件，并续传中断文件：

    export TAO_STORAGE_ROOT=/path/to/mounted-data-disk/tao-training
    python -m datasets.vpt.minestudio_download \
        --stage foundation \
        --data-root $TAO_STORAGE_ROOT/minestudio \
        --modalities action meta_info

    python -m datasets.vpt.minestudio_download \
        --stage foundation \
        --data-root $TAO_STORAGE_ROOT/minestudio \
        --modalities image \
        --image-shard-index 0 1 2 3

在正式下载前可给任一命令添加 `--list-only`，只查看解析出的模态、分片与目标目录。

使用 `--all-image-shards` 可以下载该阶段全部图像；分别对 foundation、construction、
long_horizon 执行即可预取完整主课程。三个阶段全部 image 约 642.5 GB，再加全量
action/meta_info 约 105.8 GB，应为数据与 checkpoint 预留至少约 800 GB。下载特定
Hugging Face branch、tag 或 commit 时传 `--revision REVISION`。预取多个图像分片时
不要传 `--replace-image-shards`，否则新文件完整发布后会删除本次未选中的旧图像分片。
不同 revision 建议使用不同的 `--data-root`，避免在同一目录混合两版 LMDB。
课程训练器若检测到某阶段本地已有两个或更多完整 image LMDB，会自动进入预取保护
模式，不再按当前分片删除该阶段的其他图像。

默认课程自动完成以下工作：

- 按 `7xx → 9xx → 10xx` 查询并遍历当前远端 `12 + 4 + 4` 个图像分片；
- 每次保留该阶段全部 action/meta_info 和一个正在训练的 image LMDB；
- 分配 `100k + 50k + 50k` 个 optimizer update，学习率依次为
  `1e-4 / 7e-5 / 5e-5`；物理 batch 为 2，梯度累计 4 次，有效 batch 为 8；
- 每 1,000 step 在当前分片的 episode 级 2% 留出集上评估 32 个 batch；
- 原子保存 `last.pt` 和轻量 `last.json`，记录累计 step、阶段与真实图像分片；
- 新分片完整下载后才删除同阶段旧图像；阶段 checkpoint 完成后删除已完成阶段数据，
最后一个启用阶段除外。删除的数据可从 Hugging Face 重新下载，checkpoint 不删除。

这里不按同名分片配对三种模态：image、action 和 meta_info 的 `part-*` 编号本来就
可以不同。读取器扫描各自全量 episode 索引，再以 episode 名称和帧数取交集。因此
动作与元数据绝不能跟随当前图像编号只下载一个分片；只有图像库允许滚动加载。

进程被抢占或连接中断后，使用完全相同的命令重启即可。首次运行会写入
`autodl_schedule.json`；恢复时若模型尺寸、有效 batch、课程 update 或数据顺序发生变化，
入口会拒绝错配续训。需要改变这些参数时应使用新的输出目录。

默认配置面向 24 GB 及以上显存；若首轮 CUDA 冒烟出现 OOM，使用
`--batch 1 --gradient-accumulation-steps 8` 保持有效 batch。安全轮换会短暂同时保留
新旧两个图像分片，因此建议数据盘至少预留约 150 GB，而不是按单分片稳态占用估算。

`6xx` 的记录器版本更杂，`8xx` 的从零建屋与 `9xx` 重叠，二者暂不进入主课程，
保留作后续消融。这个顺序参考了 OpenAI VPT 的“基础 BC 后再按 house / early-game
数据专门化”训练结构，而不是把不同任务数据等权混洗。

需要排查单个分片时，可手工下载当前阶段的全部动作/元数据 LMDB 和一个图像 LMDB：

    python -m datasets.vpt.minestudio_download \
        --stage foundation \
        --image-shard-index 0 \
        --data-root runs/data/minestudio

动作加元数据库常驻约 14–64 GB，单个图像分片通常为几十 GB。元数据提供逐帧
GUI 状态和绝对光标坐标，但它们
只保留为辅助监督与诊断目标，绝不直接送入策略。MineStudio 和 CraftGround 都把
GUI 光标渲染进 RGB，策略从图像感知光标，并由两轴 camera 动作监督相对鼠标移动；
GUI 窗口不会被过滤。切换分片时加入
`--replace-image-shards`，会在新分片完整下载后只删除本阶段的旧图像 LMDB；因此本地
不需要容纳整组数据。该删除不可从本项目恢复，但 Hugging Face 下载可重新执行。
当前远端结构中 `7xx / 9xx / 10xx` 分别有 `12 / 4 / 4` 个图像分片；训练日志和
checkpoint 都会记录本轮实际读取的分片名。

然后训练当前阶段：

    python -m train.minecraft.world_model_warm_start \
        --data-root runs/data/minestudio \
        --stage foundation \
        --steps 100000

同阶段下一图像分片或下一课程都从同一个 v4 checkpoint 续训；`--steps` 是全局累计
优化步数。例如先准备 `construction` 的图像分片，再执行：

    python -m train.minecraft.world_model_warm_start \
        --data-root runs/data/minestudio \
        --stage construction \
        --steps 150000 \
        --resume runs/checkpoints/minecraft_dreamer_lite/last.pt

世界模型负责动作条件化的下一视觉 latent 预测与离散 KL；奖励、继续概率、事件和
物品栏头已定义，但只有 CraftGround 回放提供对应真值后才应加入损失。训练 checkpoint
显式标记为 `minecraft_dreamer_lite_v4`；v4 移除训练期绝对光标特权输入，仅从 RGB
观察光标并保留 GUI 动作监督，不与 PixelTower、旧 v3/v2/v1 世界模型 checkpoint
静默部分加载。旧 PixelTower BC
入口仍读取原始 `mp4 + jsonl`；新的联合
训练入口直接读取 MineStudio v1.1 的 `image + action + meta_info` LMDB，不安装
MineStudio 本体及其模拟器依赖；约 400 GB segmentation 以及当前没有监督头的
event/motion 都不会下载。

## 模型规模依据

主课程三组图像约 642.5 GB，占五组 MineStudio 图像 901.6 GB 的约 71.3%。按 VPT
论文约 2,000 小时承包商数据作比例估计，主课程约 1,425 小时、1.026 亿个 20 Hz
帧。视觉 patch 高度相关，不能把每帧 576 patch 机械当成 576 个独立文本 token；
代码将每帧折算为 40 个有效多模态 token，再用 `有效 token ≈ 20 × 参数` 作为选档
启发式，得到约 205M 可训练参数。当前快塔加世界模型约 211M，与数据侧档位一致，
所以不继续放大到 300M+。冻结编码器不计入这个可训练参数预算。

这个公式只用于在 100M / 210M / 320M 档位间选择，不宣称是视觉控制的普适缩放律。
每个已下载 LMDB 的实际帧数会在训练启动时打印；最终是否增减模型，应以 AutoDL 的
吞吐、验证 BC loss 和固定 seed 的 CraftGround 闭环成功率共同决定。

数据与任务说明来源：[OpenAI VPT contractor demonstrations](https://github.com/openai/Video-Pre-Training#contractor-demonstrations)、
[MineStudio v1.1 数据接口](https://github.com/CraftJarvis/MineStudio/tree/v1.1.4/minestudio/data/minecraft)。

## 快塔 v2

`net.spatiotemporal_fast_tower.SpatiotemporalFastTower` 是不兼容旧 checkpoint 的新结构。
冻结 DINO 和冻结文本编码器在模型外运行，快塔核心不下载权重。正式默认配置约
100M 可训练参数，动作块长度 `K=4`。默认输入契约如下：

- 当前帧：`[B,576,Dv]`，对应 `18×32` 完整 patch 网格；
- 历史帧：`[B,H,576,Dv]`，内部做 `2×2` 池化后逐空间位置进行时间混合；
- 文本：完整 token 与有效位，不使用 UTF-8 字节从零学习语义；
- 控制状态：过去动作和每步 `dt`；没有结构化光标坐标输入，GUI 光标只从 RGB 感知；
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
