# Godot、CraftGround 与 MineStudio 训练项目

仓库保留两套在线环境：Godot 共享内存强化学习环境与 CraftGround 在线环境。Minecraft
离线训练只保留一条 MineStudio 路径：完整下载指定数据范围与模态，然后在全部本地数据
上无限循环训练 `SpatiotemporalFastTower + Dreamer-lite`。

## 安装

需要 Python 3.11+。Godot 环境另需 Godot 4.6.1 .NET 与 .NET 8；CraftGround 环境
另需 Java 21。Minecraft 训练需要支持 BF16 的 CUDA GPU。

    python -m pip install -e .

## MineStudio 无限训练

默认范围是 `10xx`，默认模态是完整 `image + action`：

    export TAO_STORAGE_ROOT=/root/autodl-tmp/tao-training
    export HF_HOME=$TAO_STORAGE_ROOT/huggingface
    python -m train.minecraft.world_model_training \
        --dataset-group 10xx \
        --modalities image action \
        --data-root $TAO_STORAGE_ROOT/minestudio \
        --output $TAO_STORAGE_ROOT/checkpoints

入口按固定顺序执行：

1. 从对应 CraftJarvis Hugging Face 数据仓库下载所选模态的完整数据；已有完整文件会复用，
   中断文件由 Hugging Face 缓存续传，训练器不会删除下载结果。
2. 加载冻结 DINOv3-S、MiniLM、时空快塔和潜状态世界模型。
3. 用真实训练前向、反向和 fused AdamW 状态探测 batch；在计入下一批 CUDA 预取余量后，
   选择估计峰值不超过总显存 75% 的最大 batch。
4. 在全部本地图像分片上无限循环训练，每 1,000 step 验证并原子保存 checkpoint。

`--target-memory-fraction` 可修改 0.75 目标，`--maximum-auto-batch` 限制探测上界。
`--resume auto` 是默认值，会自动恢复输出目录中的 `last.pt`。按 Ctrl+C 时会保存最近完成
的 step。需要额外读取元数据时，把 `meta_info` 加到 `--modalities`；训练必需的
`image` 和 `action` 不能省略。

当前记录的必需数据规模（十进制 GB）：

| 范围 | image | action | 合计 |
|---|---:|---:|---:|
| `7xx` | 368.636 | 17.660 | 386.296 |
| `9xx` | 178.950 | 8.049 | 186.999 |
| `10xx` | 94.908 | 4.832 | 99.740 |

只下载而不启动训练：

    python -m datasets.minestudio.download \
        --dataset-group 10xx \
        --modalities image action \
        --data-root $TAO_STORAGE_ROOT/minestudio

## 在线环境

Godot：

    export GODOT_EXE=/path/to/godot
    python -m rl_training_environments.godot.train_ppo --total-timesteps 100000

Godot 像素训练需要真实 X11/Vulkan 渲染，不能用 `--headless` 哑渲染器。协议说明见
`rl_training_environments/godot/engine/README.md`。

CraftGround 的环境、奖励塑形、动作契约、回放和世界快照位于
`rl_training_environments/craftground/`，本轮没有删除或改写这些文件。

## 目录

    datasets/minestudio/                   MineStudio 完整下载与 LMDB 读取
    blocks/                                通用注意力、调制、残差与 Transformer 算子
    net/                                   时空快塔与训练期潜状态世界模型
    train/minecraft/                       无限联合训练入口与动作监督
    rl_training_environments/godot/        Godot 环境、SB3 适配与引擎工程
    rl_training_environments/craftground/  CraftGround 环境、回放与世界快照
    tests/                                 保留路径的契约测试
