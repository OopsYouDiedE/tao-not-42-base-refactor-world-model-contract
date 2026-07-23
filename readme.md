# Godot 与 CraftGround 训练项目

仓库保留两套在线环境：Godot 共享内存强化学习环境与 CraftGround 在线 Minecraft
环境。动作以 `net/action_token_codec.py` 的 `StructuredAction`（CraftGround V2 键序 +
相机 mu-law 分箱）为唯一规范表示，结构上强制互斥约束合法。以 Gemma 4（MoE VLM）为
视觉主干、直接自回归生成动作 token 的 VLA 策略定义在 `net/gemma4_policy.py`。

Minecraft 侧的**行为数据来源**是 `data_pipelines/mineflayer_actions/`：用 mineflayer
驱动无头 bot 在真实 Java 服务器上主动执行动作，记录每个动作的起始 tick 与持续时长，
并在关键节点截取观测帧，产出 observation(t)→action(t) 的图文配对数据。旧的 MineStudio
离线下载 / LMDB 读取 / VPT 动作编码路径，以及依赖它的离线 SFT 训练入口已整体移除。

## 一键配置环境与验证

需要 Git、Python 3.11+。Godot 环境另需 Godot 4.6.1 .NET 与 .NET 8；CraftGround
环境另需 Java 21。下面这一段可整体复制到空目录执行，完成克隆、建虚拟环境、安装、
契约测试与全量编译校验：

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
    python -m compileall -q blocks data_pipelines net rl_training_environments train tests

## 在线环境

Godot：

    export GODOT_EXE=/path/to/godot
    python -m rl_training_environments.godot.train_ppo --total-timesteps 100000

Godot 像素训练需要真实 X11/Vulkan 渲染，不能用 `--headless` 哑渲染器。协议说明见
`rl_training_environments/godot/engine/README.md`。

CraftGround 的环境、奖励塑形、动作契约、回放和世界快照位于
`rl_training_environments/craftground/`。

## Minecraft 行为数据来源

`data_pipelines/mineflayer_actions/` 用 mineflayer 驱动无头 bot 在真实 Java Minecraft
服务器上主动执行动作，记录每个动作的起始 tick 与持续时长，覆盖移动 `F/B/L/R`、姿态
`jump/sneak/sprint`、转视角 `cam(dYaw,dPitch)`、合成 `craft:*`、放置 `use`、破坏
`attack` 六类。它还能在每个动作节点截取第一人称观测帧，产出 observation→action 图文
配对。从零搭建服务器到采集的完整流程见 `data_pipelines/mineflayer_actions/SETUP.md`，
动作字段与原理见 `data_pipelines/mineflayer_actions/AGENTS.md`。

时间基准是 `bot.time.age`（世界年龄，20 tick/秒）。该子包是 Node.js 工具链（依赖见其
`package.json`），产物（server.jar、世界、node_modules、采集 JSON/PNG）均为运行期数据，
不入库。

## 目录

    data_pipelines/mineflayer_actions/     mineflayer 主动执行动作与观测帧采集
    blocks/                                通用注意力、调制、残差与 Transformer 算子
    net/                                   Gemma4 动作策略与动作 token 编解码
    train/minecraft/                       数据源无关的关键动作评估指标
    rl_training_environments/godot/        Godot 环境、SB3 适配与引擎工程
    rl_training_environments/craftground/  CraftGround 环境、回放与世界快照
    tests/                                 保留路径的契约测试
