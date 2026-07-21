# Godot、CraftGround 与 MineStudio 训练项目

仓库保留两套在线环境：Godot 共享内存强化学习环境与 CraftGround 在线环境。Minecraft
离线训练只保留一条 MineStudio 路径：完整下载指定数据范围与模态，然后在全部本地数据
上无限循环用 LoRA 对 **Gemma 4（MoE VLM）**做行为克隆(SFT)。策略以 Gemma4 为视觉
主干，把历史帧、任务文本与过去动作组成多模态 prompt，**自回归直接生成动作 token**，
再由 `net/action_token_codec.py` 解码成结构合法的低层控制动作。旧的
`SpatiotemporalFastTower + Dreamer-lite` 快慢塔路径已移除。

## 一键全流程：配置环境 → 验证 → 训练

需要 Git、Python 3.11+。Godot 环境另需 Godot 4.6.1 .NET 与 .NET 8；CraftGround
环境另需 Java 21。Minecraft 训练需要支持 BF16 的 CUDA GPU（gemma-4-26B-A4B 推理约需 17GB
显存）。Gemma4 权重公开、无需接受条款；上传 checkpoint 仍需一个具有写权限的 Hugging
Face token。下面这一段可整体复制到空目录
执行，依次完成克隆、建虚拟环境、安装、契约测试与全量编译校验，最后启动无限训练；因为
训练是无限循环，测试与编译校验必须放在训练之前，训练一旦启动其后的命令不会再执行：

    # 1) 克隆 + 虚拟环境 + 安装(含开发依赖 pytest)
    git clone \
        https://github.com/OopsYouDiedE/tao-not-42-base-refactor-world-model-contract.git
    cd tao-not-42-base-refactor-world-model-contract
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -e ".[dev]"

    # 2) 验证安装:契约测试 + 全量编译(与 AGENTS.md 的门禁一致)
    python -m pytest
    python -m compileall -q blocks datasets net rl_training_environments train tests

    # 3) 配置存储与 Hugging Face 登录(token 不进命令参数,只用登录态)
    export TAO_STORAGE_ROOT="${TAO_STORAGE_ROOT:-$PWD/runs}"
    export HF_HOME="$TAO_STORAGE_ROOT/cache/huggingface"
    export HF_REPO_ID=unjustify/minecraft-gemma4-vla-10xx
    mkdir -p "$TAO_STORAGE_ROOT" "$HF_HOME"
    hf auth whoami >/dev/null 2>&1 || hf auth login

    # 4) 下载所选数据范围(默认 10xx,约 100GB)并无限 LoRA SFT
    python -m train.minecraft.world_model_training \
        --dataset-group 10xx \
        --modalities image action \
        --data-root "$TAO_STORAGE_ROOT/data/minestudio" \
        --cache-directory "$HF_HOME/hub" \
        --output "$TAO_STORAGE_ROOT/checkpoints/minecraft-gemma4-vla-10xx" \
        --hub-repo-id "$HF_REPO_ID"

第 4 步默认范围是 `10xx`，默认模态是完整 `image + action`。做实验或联调时加
`--max-image-shards 1` 只下载 1 个图像分片（几 GB）即可端到端跑通。换其它数据范围继续
训练时，在已有 checkpoint 基础上加 `--allow-dataset-transfer` 放开 dataset-group 校验
（策略结构仍要求严格一致）。闭环成功率验收是单独一步（需真实 Java Minecraft 与显示
服务），见下文“有效性验收”。

入口按固定顺序执行：

1. 从对应 CraftJarvis Hugging Face 数据仓库下载所选模态的完整数据；已有完整文件会复用，
   中断文件由 Hugging Face 缓存续传，训练器不会删除下载结果。
2. 加载 gemma-4-26B-A4B（bf16，冻结主干）并注入 LoRA 适配器，只训练适配器参数。
3. 逐窗口构造多模态 prompt 与目标动作 token 文本，用梯度累积做行为克隆 SFT。
4. 在全部本地图像分片上无限循环训练，每 1,000 step 验证并原子保存 LoRA 适配器
   checkpoint；`last.pt` 和 `last.json` 随后在后台上传到指定的公开 Hugging Face 模型仓库。

Gemma4 权重公开、匿名即可下载；只有上传 checkpoint 需要写权限 token，用 `hf auth login`
登录态认证，不把 token 放入命令参数。若 `--hub-repo-id` 不存在，入口会创建公开模型仓库；
若同名仓库已经是私有仓库，入口会拒绝上传，避免继续占用私有仓库额度。

数据正文位于 `$TAO_STORAGE_ROOT/data/minestudio/10xx`，数据续传元数据位于该目录内的
`.cache/huggingface`。Gemma4 权重的共享缓存由 `--cache-directory` 指定，上例为
`$HF_HOME/hub`。本地 checkpoint 位于
`$TAO_STORAGE_ROOT/checkpoints/minecraft-gemma4-vla-10xx`。`TAO_STORAGE_ROOT` 未设置时
默认使用当前仓库的 `runs/`；需要放到其他磁盘时，在运行前自行设置绝对路径即可。

`--action-format` 选择动作 token 文本格式（`compact_tag` / `key_value` / `json_line`），
`--action-horizon` 是一次生成的未来帧数，`--lora-rank` 控制适配器容量，
`--gradient-accumulation` 是等效 batch。`--resume auto` 是默认值，会自动恢复输出目录中的
`last.pt`。按 Ctrl+C 时会保存最近完成的 step。训练必需的 `image` 和 `action` 不能省略。

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

训练器每次验证会在 episode 级固定留出集上输出 SFT loss 与贪心生成的**关键动作
一致率**（只比移动 / 转向 / 姿态 / 攻击 / 使用 / 快捷栏 / 相机粗方向，抖动级差异不计）。
正式验收包含以下两层：

1. 在固定 seed、按 episode 隔离且从未参与训练的留出集上，报告关键动作一致率，
   与随机历史、无历史等对照条件比较，判断策略是否真正利用了观测与历史。
2. 在 CraftGround 固定任务、固定 seed 上做闭环回放，报告样本量、成功率及 95% 置信
   区间，并与 no-op 基线比较。离线指标通过且闭环成功率稳定领先基线，才能判定训练足够
   有效；连续多个 checkpoint 不再改善时再考虑停止。

第二层由闭环入口直接执行。下面默认使用固定 seed `0..9`，每局最多 12,000 tick，成功
条件是库存曾出现 `diamond_pickaxe`；同一组 seed 还会运行 no-op 基线。当 checkpoint 的
Wilson 95% 下界高于 no-op 上界时，`effective_over_noop_95` 才为 `true`：

    python -m train.minecraft.evaluate_checkpoint \
        --checkpoint "$TAO_STORAGE_ROOT/checkpoints/minecraft-gemma4-vla-10xx/last.pt" \
        --dataset-group 10xx \
        --cache-directory "$HF_HOME/hub" \
        --seeds 0 1 2 3 4 5 6 7 8 9 \
        --maximum-steps 12000

闭环评估会启动真实 Java Minecraft，需要可用的显示服务。策略一次生成动作块并逐 tick
执行，队列耗尽后重新规划。只想跳过耗时加倍的 no-op 对照时可加 `--no-compare-noop`，
但这种结果不能产生对基线的保守判定。

## 动作 token 表示实验

在投入长时间 SFT 前，可先用 `action_token_probe` 探测 Gemma4 基座最擅长哪种动作 token
表示。它随机抽取数据窗口，对每种格式 × horizon（默认 5 与 20）× 上下文条件（正确历史 /
无历史 / 随机历史）让模型重复生成并解码，用关键动作一致率与自一致性评分，输出 JSON 行
与 markdown 报告：

    python -m train.minecraft.action_token_probe \
        --data-directory "$TAO_STORAGE_ROOT/data/minestudio/10xx" \
        --formats compact_tag key_value json_line \
        --horizons 5 20 --windows 3 --repetitions 8 \
        --report "$TAO_STORAGE_ROOT/action_token_probe/report.md" \
        --json-log "$TAO_STORAGE_ROOT/action_token_probe/records.jsonl"

## 在线环境

Godot：

    export GODOT_EXE=/path/to/godot
    python -m rl_training_environments.godot.train_ppo --total-timesteps 100000

Godot 像素训练需要真实 X11/Vulkan 渲染，不能用 `--headless` 哑渲染器。协议说明见
`rl_training_environments/godot/engine/README.md`。

CraftGround 的环境、奖励塑形、动作契约、回放和世界快照位于
`rl_training_environments/craftground/`，本轮没有删除或改写这些文件。

## 目录

    data_pipelines/minestudio/                   MineStudio 完整下载与 LMDB 读取
    blocks/                                通用注意力、调制、残差与 Transformer 算子
    net/                                   Gemma4 动作策略与动作 token 编解码
    train/minecraft/                       无限 LoRA SFT 入口、动作监督与动作 token 实验
    rl_training_environments/godot/        Godot 环境、SB3 适配与引擎工程
    rl_training_environments/craftground/  CraftGround 环境、回放与世界快照
    tests/                                 保留路径的契约测试
