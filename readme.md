# Godot、CraftGround 与 MineStudio 训练项目

仓库保留两套在线环境：Godot 共享内存强化学习环境与 CraftGround 在线环境。Minecraft
离线训练只保留一条 MineStudio 路径：完整下载指定数据范围与模态，然后在全部本地数据
上无限循环训练 `SpatiotemporalFastTower + Dreamer-lite`。

## 安装

需要 Git、Python 3.11+。Godot 环境另需 Godot 4.6.1 .NET 与 .NET 8；CraftGround
环境另需 Java 21。Minecraft 训练需要支持 BF16 的 CUDA GPU。从空目录安装：

    if ! command -v git >/dev/null 2>&1; then
        echo "请先按当前操作系统安装 Git"
    else
        repository_ready=false
        if [ -d tao-not-42-base-refactor-world-model-contract/.git ]; then
            cd tao-not-42-base-refactor-world-model-contract && repository_ready=true
        elif git clone \
            https://github.com/OopsYouDiedE/tao-not-42-base-refactor-world-model-contract.git && \
            cd tao-not-42-base-refactor-world-model-contract; then
            repository_ready=true
        fi
        if $repository_ready; then
            python -m venv .venv && \
                source .venv/bin/activate && \
                python -m pip install --upgrade pip && \
                python -m pip install -e .
        fi
    fi

## MineStudio 无限训练

默认范围是 `10xx`，默认模态是完整 `image + action`：

    export TAO_STORAGE_ROOT="${TAO_STORAGE_ROOT:-$PWD/runs}"
    export HF_HOME="$TAO_STORAGE_ROOT/cache/huggingface"
    export HF_REPO_ID=unjustify/minecraft-dreamer-lite-10xx
    mkdir -p "$TAO_STORAGE_ROOT" "$HF_HOME"
    hf auth whoami >/dev/null 2>&1 || hf auth login
    python -m train.minecraft.world_model_training \
        --dataset-group 10xx \
        --modalities image action \
        --data-root "$TAO_STORAGE_ROOT/data/minestudio" \
        --cache-directory "$HF_HOME/hub" \
        --output "$TAO_STORAGE_ROOT/checkpoints/minecraft-dreamer-lite-10xx" \
        --hub-repo-id "$HF_REPO_ID"

入口按固定顺序执行：

1. 从对应 CraftJarvis Hugging Face 数据仓库下载所选模态的完整数据；已有完整文件会复用，
   中断文件由 Hugging Face 缓存续传，训练器不会删除下载结果。
2. 加载冻结 DINOv3-S、MiniLM、时空快塔和潜状态世界模型。
3. 用真实训练前向、反向和 fused AdamW 状态探测 batch；在计入下一批 CUDA 预取余量后，
   选择估计峰值不超过总显存 75% 的最大 batch。
4. 在全部本地图像分片上无限循环训练，每 1,000 step 验证并原子保存 checkpoint；
   `last.pt` 和 `last.json` 随后在后台上传到指定的公开 Hugging Face 模型仓库。

运行前需要在 DINOv3 模型页接受访问条款，并用具有写权限的 token 执行 `hf auth login`。
代码使用登录态认证，不把 token 放入命令参数。若 `--hub-repo-id` 不存在，入口会创建公开
模型仓库；若同名仓库已经是私有仓库，入口会拒绝上传，避免继续占用私有仓库额度。

数据正文位于 `$TAO_STORAGE_ROOT/data/minestudio/10xx`，数据续传元数据位于该目录内的
`.cache/huggingface`。DINOv3 与 MiniLM 的共享缓存由 `--cache-directory` 指定，上例为
`$HF_HOME/hub`。本地 checkpoint 位于
`$TAO_STORAGE_ROOT/checkpoints/minecraft-dreamer-lite-10xx`。`TAO_STORAGE_ROOT` 未设置时
默认使用当前仓库的 `runs/`；需要放到其他磁盘时，在运行前自行设置绝对路径即可。

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
        --data-root "$TAO_STORAGE_ROOT/data/minestudio" \
        --cache-directory "$HF_HOME/hub"

## 有效性验收

训练器每次验证会在 episode 级固定留出集上同时输出 loss、相机角度 MAE、动作
macro-F1、稀有按钮 recall、完整动作准确率、no-op 准确率基线，以及真实动作和打乱动作
在每个 horizon 的 open-loop 世界模型误差。正式验收包含以下三层：

1. 在固定 seed、按 episode 隔离且从未参与训练的留出集上，报告相机角度 MAE、互斥
   动作块 macro-F1、稀有按钮 recall 和整组动作准确率，并与多数类、动作持久化基线比较。
2. 对世界模型按未来第 1 至第 4 步分别报告误差；再将真实动作替换为打乱动作。只有真实
   动作误差显著更低，才能证明模型实际利用动作，而不是潜状态塌缩或只复制上一帧。
3. 在 CraftGround 固定任务、固定 seed 上做闭环回放，报告样本量、成功率及 95% 置信
   区间，并与未训练模型和纯行为克隆基线比较。离线指标通过且闭环成功率稳定领先基线，
   才能判定训练足够有效；连续多个 checkpoint 不再改善时再考虑停止。

第三层由闭环入口直接执行。下面默认使用固定 seed `0..9`，每局最多 12,000 tick，成功
条件是库存曾出现 `diamond_pickaxe`；同一组 seed 还会运行 no-op 基线。当 checkpoint 的
Wilson 95% 下界高于 no-op 上界时，`effective_over_noop_95` 才为 `true`：

    python -m train.minecraft.evaluate_checkpoint \
        --checkpoint "$TAO_STORAGE_ROOT/checkpoints/minecraft-dreamer-lite-10xx/last.pt" \
        --dataset-group 10xx \
        --cache-directory "$HF_HOME/hub" \
        --seeds 0 1 2 3 4 5 6 7 8 9 \
        --maximum-steps 12000

闭环评估会启动真实 Java Minecraft，需要可用的显示服务。只想先运行 checkpoint、跳过
耗时加倍的 no-op 对照时可加 `--no-compare-noop`，但这种结果不能产生对基线的保守判定。
若有旧版、未训练初始化或其他消融 checkpoint，可在评估命令追加
`--baseline-checkpoint /path/to/baseline/last.pt`；结果会额外输出
`effective_over_baseline_95`。当前快塔动作头本身使用 BC 监督，而世界模型是独立训练期
辅助路径，所以关闭世界模型 loss 不能构成有意义的“非 BC 对照”。

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
