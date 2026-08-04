# TAO

本仓库按在线环境交互、数据适配、行为克隆、轨迹审核、模型判断审核、相对优势训练和环境验证的职责边界组织。行为克隆与 2+6 相对优势训练入口已经恢复，统一训练标准输入动作协议 v1。

旧的 `tao/`、`dataset/`、`train/`、`game_environment/`、`tools/` 和 `scripts/` 入口不再代表当前工作树。训练入口只存在于当前职责目录中。

## 当前能力

| 能力 | 入口 | 状态 |
| --- | --- | --- |
| 标准输入动作协议 | `online_interactive_environments/STANDARD_INPUT_ACTION_PROTOCOL.md` | v1 合同、校验和 CraftGround 执行适配可用 |
| 动作序列解析与逐 tick 调度 | `online_interactive_environments.ActionSequenceCompiler` | 可用 |
| 生成记录与模型延迟指标 | `ActionSequenceCompiler(record_generations=True)` | 可用；模型墙钟延迟不换算为环境 tick |
| CraftGround 环境创建 | `online_interactive_environments.craftground.create_environment` | 可用 |
| CraftGround 内存快照 | `MemorySnapshotCoordinator` | 可用 |
| 多环境并行推演 | `ParallelRolloutRunner` | 可用 |
| Godot 聚光灯强化学习环境 | `online_interactive_environments.godot` | 已恢复；需要 Godot 4.6 .NET、NumPy、Gymnasium 和 Stable-Baselines3 |
| 协议动作到 CraftGround 动作的适配 | `CraftGroundKeyboardMouseAdapter` | 可用 |
| 四分支同策略轨迹 | `environment_validation_tools.run_four_teacher_trajectories` | 可用；限定 WSL 2 Ubuntu-24.04 |
| 轨迹审核与相对优势比较 | `interaction_trajectory_review_agents`、`relative_advantage_comparison_training` | 可用 |
| 比较结论复核 | `model_judgment_review_agents` | 可用 |
| 视觉行为克隆 | `python -m behavior_cloning_training.train` | 已恢复；需要 GPU、Unsloth、TRL 和模型依赖 |
| 2+6 策略 RLHF | `python -m relative_advantage_comparison_training.train_policy` | 已恢复；要求生成时记录 on-policy token logprob |

## 协议与执行后端

项目当前动作文本协议统一命名为 **标准输入动作协议 v1**。协议版本描述文本格式、tick、设备和输入语义。

CraftGround Python 包中的 `ActionSpaceVersion.V2_MINERL_HUMAN` 是上游枚举名称。本项目将该执行后端统一称为 **`keyboard_and_mouse_only`**。它表示 CraftGround 当前只接收键盘和鼠标动作，不是本项目的 v2 协议，也不表示标准输入动作协议已经升级到 v2。

两者关系如下：

| 层级 | 当前名称 | 含义 |
| --- | --- | --- |
| 文本协议 | 标准输入动作协议 v1 | 模型输出的设备、tick 和动作文本合同 |
| 项目执行后端 | `keyboard_and_mouse_only` | 当前允许接入 CraftGround 的设备范围 |
| CraftGround 上游枚举 | `V2_MINERL_HUMAN` | 创建环境时必须传递的第三方 API 标识 |

编译器把设备信息写入 `TickDecision`，`TeacherTrajectoryExecutor` 校验 `KeyboardMouse` 设备并通过
`CraftGroundKeyboardMouseAdapter` 转换为完整的 CraftGround V2 动作字典。每个已接受协议 tick 对应
一次 `environment.step()`。后端声明见 `online_interactive_environments/CRAFTGROUND_KEYBOARD_AND_MOUSE_ONLY_BACKEND.md`。

## 代码结构

| 路径 | 职责 | 当前内容 |
| --- | --- | --- |
| `external_dataset_loaders_and_protocol_adapters/` | 外部数据集加载、预处理和项目协议适配 | 迁移占位 |
| `behavior_cloning_dataset_converters/` | 行为克隆数据集转换 | 迁移占位 |
| `behavior_cloning_training/` | 标准输入动作协议视觉行为克隆 | JSONL/HDF5 conversation、MineStudio 流式数据和 LoRA SFT |
| `online_interactive_environments/` | 在线环境实现、环境配置和环境协议 | 动作编译器、CraftGround 运行时、快照、并行推演和环境协议 |
| `online_interactive_environments/godot/` | Godot 在线强化学习环境 | 40 环境共享内存通信、Godot 引擎工程、SB3 向量环境和 PPO 训练入口 |
| `interaction_trajectory_review_agents/` | 交互轨迹审核代理 | 协议、预算、生成状态和任务结果审核 |
| `online_environment_interaction_agents/` | 在线环境交互代理、动作生成、执行和轨迹记录 | 在线轨迹生成 Agent 提示词结构 |
| `model_judgment_review_agents/` | 模型判断结果审核代理 | 比较均值、排序和选择结论复核 |
| `relative_advantage_comparison_training/` | 相对优势样本、计算和训练 | 同起点 2+6 样本、on-policy 采样和 clipped LoRA 更新 |
| `environment_validation_tools/` | 环境接口、协议链路和项目结构验证 | 项目结构校验工具 |
| `shared_tools/` | 跨职责共享且无领域语义的基础设施 | 环境变量与 `.env` 配置读取 |
| `tests/` | 当前已迁移模块的自动化测试 | 动作编译器、CraftGround 组件和项目结构测试 |

旧入口仍然不可用；BC 与 RLHF 必须通过当前职责目录运行。

## BC 与 RLHF

行为克隆读取标准协议 v1 的 JSONL/HDF5 conversation，或直接从 MineStudio LMDB 流式读取：

```bash
python -m behavior_cloning_training.train \
  --model unsloth/gemma-4-26B-A4B-it \
  --dataset runs/datasets/minestudio-data-10xx-v110 \
  --streaming \
  --output runs/training/bc
```

RLHF execution group 必须包含同一起点的 2 条 `reference_expert` 与 6 条 `policy_sample`；
policy 样本必须携带生成时的 `response_token_ids`、`old_logprobs`、`policy_version` 和采样参数：

```bash
python -m relative_advantage_comparison_training.train_policy \
  --model unsloth/gemma-4-26B-A4B-it \
  --adapter runs/training/bc/lora_adapter \
  --execution runs/rlhf/iteration-0001/execution.json \
  --intent "Approach the visible tree" \
  --output runs/training/rlhf-iteration-0001
```

这些训练入口需要在 GPU Linux 环境验收；离线测试不能替代真实训练结论。

在 GPU Linux 上可用一条命令运行 2B 视觉模型的真实推理、单步 BC 和单轮 2+6 相对优势训练：

```bash
python -m environment_validation_tools.run_gpu_training_validation \
  --output runs/gpu_training_validation/qwen3-vl-2b
```

该命令即时构造公开合同数据，并在策略生成阶段使用显式记录的合法协议候选集约束解码。它用于验证
本地模型训练链路，不替代 CraftGround 环境轨迹的任务成功验收。

## 安装

默认 GPU Linux 服务器使用以下一条命令安装完整 CraftGround 与训练环境：

```bash
bash scripts/bootstrap_gpu_craftground.sh
```

不需要 CraftGround 时使用通用安装脚本。项目提供锁定版本和最新兼容版本两种方式，脚本按已安装
PyTorch 的真实 CUDA 可用性自动选择 CPU 或 CUDA；CPU 路径不会安装 Unsloth、Flash Attention、
xFormers 或 CUDA runtime 等 GPU 专用包：

```bash
bash scripts/bootstrap.sh
bash scripts/bootstrap.sh --latest
```

两个脚本都在安装后校验 Python 版本、核心依赖导入和目标计算后端。项目源码不包含环境检查与鉴权
检查模块。本地 GitHub 和 Hugging Face 按需自行执行 `gh auth login` 与 `hf auth login`；教师模型
API 通过根目录 `.env` 或进程环境变量配置。完整安装、鉴权与配置合同见 `docs/INSTALLATION.md`。

创建真实 CraftGround 环境还需要 JDK 21。CraftGround Python 包与 mc121 runtime 均锁定到
`OopsYouDiedE/CraftGround` 的同一精确提交；首次创建环境时会复制维护版 runtime 并执行 Gradle 构建。

## 动作编译器

```python
from online_interactive_environments import ActionSequenceCompiler

compiler = ActionSequenceCompiler(record_generations=True)
compiler.begin_generation(provider="openai", model="model-name")

compiler.feed("Device KeyboardMouse\nTick 0\n<action>W ; MouseMove 12 -4 ; NoOp</action>")

record = compiler.end_generation()
decision = compiler.pull()
```

`record_generations` 在编译器创建时确定。启用后，每次生成按开始顺序保存完整输入分片、提交结果、首段内容时间、首个可执行动作时间和等待 tick 指标。

## CraftGround

默认入口会从已安装的维护版 CraftGround 准备构建模板，为每个环境复制独立可写 runtime，并创建
`keyboard_and_mouse_only` 后端环境。环境默认使用共享内存 IPC 和固定端口，避免 SocketIPC 的全局
孤儿进程扫描影响并行实例：

```python
from online_interactive_environments.craftground import create_environment

environment = create_environment()
observation, info = environment.reset(options={"fast_reset": False})
```

也可以使用已经准备好的 runtime：

```python
environment = create_environment(
    runtime_path="C:/craftground/maintained-runtime",
)
```

CraftGround fork、依赖锁定、快照和并行推演说明见
`online_interactive_environments/craftground/README.md`。

## 四分支轨迹

四分支入口默认从 `TEACHER_BACKEND` 选择一个后端，四个 arm 复用同一后端配置与模型。API 配置使用
`TEACHER_API_URL`、`TEACHER_API_KEY` 和 `TEACHER_MODEL`；CLI 配置使用 `TEACHER_CLI_EXECUTABLE`、
`TEACHER_MODEL` 和可选的 `TEACHER_CLI_ARGUMENTS`。入口不读取 `~/.codex/auth.json` 或
`~/.claude/settings.json`。

```bash
export TEACHER_BACKEND=openai-api
export TEACHER_API_URL=https://example.invalid/v1
export TEACHER_API_KEY=replace-me
export TEACHER_MODEL=model-name
python -m environment_validation_tools.run_four_teacher_trajectories --output runs/four-arm
```

该入口会冻结昼夜、天气、生物生成和随机 tick，清除普通实体，验证四实例玩家状态一致，按玩家实际
位置计算快照区域，并在正式推演前执行一次“扰动 → 倒档 → 状态比对”。输出包含轨迹、逐轨迹审核、
相对优势比较样本和比较结论复核。

`--trajectory-count` 控制 arm 数，`--environment-count` 控制并行 CraftGround 环境槽位数。内存不足
以为每个 arm 各开一个客户端时调小槽位数；超额 arm 由 `ParallelRolloutRunner` 在环境池外排队。

WSL 上「轨迹完全由云端模型生成 + 统一相对优势评估」的封装入口、`responses` wire 要求、内存槽位
建议和已知评分缺陷见 [`docs/CLOUD_RELATIVE_ADVANTAGE_RUN.md`](docs/CLOUD_RELATIVE_ADVANTAGE_RUN.md)：

```bash
export TEACHER_API_KEY=replace-me
TRAJECTORY_COUNT=8 ENVIRONMENT_COUNT=2 MAX_GENERATIONS=10 \
  bash scripts/run_wsl_cloud_relative_advantage.sh
```

## 验证

检查重组目录和 README 路径合同：

```bash
python -m environment_validation_tools.validate_project_structure
```

运行测试和静态检查：

```bash
python -m pytest -q
python -m ruff check .
```

## 运行产物

环境执行产生的轨迹、观察帧、截图、模型生成记录和结果报告统一写入项目根目录的
`runs/`。该目录不属于源码，不进入版本控制，可以在每轮验证结束后整体清理。
模块目录中不得创建 `test_runs/`、`runs/` 或其他持久化运行产物目录。
