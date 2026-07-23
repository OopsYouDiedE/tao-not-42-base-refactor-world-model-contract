# 项目开发规范

本文件约束本仓库中的代码、测试、文档和自动化修改。规则适用于人类开发者与 AI
助手。子目录存在 `AGENTS.md` 时，子目录规则在其范围内追加生效。

## 1. 当前范围

仓库只维护以下生产路径：

- Godot 强化学习环境及其共享内存、SB3 适配与 PPO 入口；
- CraftGround 在线环境、奖励塑形、动作契约、回放与世界快照；
- mineflayer 主动执行动作与观测帧采集的 Minecraft 行为数据来源；
- 以 Gemma4 为视觉大模型、直接生成动作 token 的 VLA 策略（`net/gemma4_policy.py`
  与 `net/action_token_codec.py`）及数据源无关的关键动作评估指标。

MineStudio 完整下载 / LMDB 读取 / VPT 动作编码、依赖它的离线 SFT 训练入口、VPT
`mp4 + jsonl`、PixelTower、VPT 教师、慢塔、判官、GRPO、MineRL、旧 DINO 地图快塔、
已移除的 `SpatiotemporalFastTower` / Dreamer-lite 世界模型和 `godot-python` 不属于当前
范围。新增这些能力必须由用户明确要求，不能根据历史文件名自行恢复。

## 2. 目录职责

按代码“是什么”放置，不按实验名称放置：

| 目录 | 职责 |
|---|---|
| `blocks/` | 与任务无关的可复用神经网络算子 |
| `net/` | 模型结构与纯配置对象，不读取文件，不启动环境 |
| `data_pipelines/` | 可跨训练流程复用的数据读取与原始数据契约 |
| `data_pipelines/mineflayer_actions/` | mineflayer 主动执行动作与观测帧采集（Node.js） |
| `rl_training_environments/godot/` | Godot 通信、进程管理、SB3 适配与训练入口 |
| `rl_training_environments/godot/engine/` | Godot 4.6.1 .NET 工程与场景 |
| `rl_training_environments/craftground/` | CraftGround 在线环境及运行状态管理 |
| `train/minecraft/` | 数据源无关的关键动作一致率与闭环置信区间指标 |
| `tests/unit/` | 不启动真实环境的纯单元测试 |
| `tests/integration/` | 跨模块契约与离线回放测试 |
| `runs/` | 数据、日志与 checkpoint，必须保持 Git ignored |

依赖方向：

```text
blocks ← net ← train
data_pipelines ← train
rl_training_environments ← train
tests 依赖上述模块，但生产代码不得 import tests
```

不同在线环境不得引用对方的具体实现。共享的任务无关逻辑应下沉到合适的公共层。

## 3. 命名规范

- 文件、目录、模块、公开类型、函数和普通变量使用完整、描述性的英文单词。
- Python 与 GDScript 文件使用 `snake_case`；C# 类型及对应文件使用 `PascalCase`。
- PPO、VPT、RL、RGB、DINO、SB3、API 等行业标准缩写可以保留。
- 禁止使用项目私有缩写，例如新写 `env`、`vec`、`mem`、`cfg`、`proc` 作为公开名称。
- 张量公式中的 `B/T/H/W/C` 是数学符号，可以保留。
- 重命名文件或公开接口时，必须同步所有 import、脚本入口、场景路径、文档和测试。

## 4. 依赖规则

- 依赖必须由生产代码中的实际 import 或引擎工程文件证明。
- Python 依赖统一声明在 `pyproject.toml`，禁止同时维护内容重复的 requirements 文件。
- 缺少生产依赖时直接报告，不在生产代码中加入 `try/except` Mock 或静默降级。
- 禁止添加 `godot-python`。Godot 与 Python 通过文件后端 mmap 通信。
- 禁止猜测 Linux 发行版包名。Godot、Java、显示服务和驱动的系统安装按运行机器处理。
- `data_pipelines/` 是本项目的顶层 Python 包；新增 HuggingFace `datasets` 依赖前必须先解决包名冲突。

## 5. 模型与数值不变量

- 除法和归一化分母必须 `clamp(min=epsilon)`，`epsilon >= 1e-4`。
- softmax、损失归约、几何计算和其他危险算子使用 fp32。
- 禁止无界 `exp`；确需指数时必须证明输入和输出有界。
- 动作必须在结构上有界。互斥动作组（前后、左右、姿态、hotbar）在解码端强制至多一个
  被激活，不能同时输出前后同按、左右同按、多个 hotbar 或潜行与冲刺同按；无论大模型
  生成的文本多脏，`net/action_token_codec.py` 的解码结果都必须结构合法。
- rollout 和递归路径不用 BatchNorm。
- 动作 token 解码永远返回定长、合法的动作块：识别不足补 noop，超出截断，非法组合消解。

## 6. 数据与训练边界

- mineflayer 采集产物（server.jar、世界、node_modules、JSON/PNG）均为运行期数据，
  存放在 `runs/` 或仓库外，不得入库；仓库内仅保留 `sample/` 下的少量参考样本。
- `data_pipelines/mineflayer_actions/` 只负责主动执行动作与观测采集，不包含优化器或
  策略训练循环。
- CraftGround 执行动作契约属于 `rl_training_environments/craftground/`。
- 离线指标不是闭环能力结论。闭环结论必须报告固定 seed、样本量和成功指标。
- checkpoint 结构不兼容时必须显式升级名称或版本，禁止静默部分加载。

## 7. Godot 协议

- Python 与 C# 的图像、元数据、动作、offset、dtype 和 seqlock 序号是硬契约。
- 任一侧修改协议时，必须同步另一侧、协议测试、Godot README 和调用链文档。
- Godot 收到 Python 动作应答前不得继续步进，保证观测帧严格消费一次。
- 图像训练需要真实渲染输出。Godot `--headless` 哑渲染器不能用于像素正确性验证。
- Godot 侧附加规则见 `rl_training_environments/godot/engine/AGENTS.md`。

## 8. 代码与文档

- 项目 Markdown 使用中文；标识符使用英文。
- 文件头 docstring 写一句职责和对外接口，不记录修改历史。
- 类和函数 docstring 使用 NumPy 或 Google 风格；张量参数与返回值声明 Shape、Dtype 和单位。
- import 位于文件顶部，禁止 `from module import *`。
- 生产代码不得包含测试 Mock、离线假数据或本地个人绝对路径。
- 废弃模块物理删除，不保留空文件、转发壳或仅含注释的兼容层。

## 9. 测试与验收

修改后按影响范围执行：

```bash
python -m pytest
python -m compileall -q blocks data_pipelines net rl_training_environments train tests
```

涉及 Godot 工程时还要执行：

```bash
dotnet build rl_training_environments/godot/engine/GodotMetaReinforcementLearning.csproj --nologo
```

并使用 Godot 4.6.1 .NET 编辑器解析 `rl_training_environments/godot/engine/`。涉及
真实渲染、CUDA 或 Java 进程的修改，纯 CPU 测试不能替代对应环境冒烟。

测试失败时必须说明失败命令和直接原因。不得把“缺少本地依赖”描述为代码测试通过。

## 10. Git 与修改纪律

- 保留用户已有的未提交修改；不得用 `git reset --hard` 或覆盖式 checkout 清理工作树。
- 删除、移动和批量重命名前先核对精确目标，并同步所有引用。
- 每轮完成的修改创建一个职责单一的 Git commit。
- commit message 使用中文说明结果，例如 `refactor: 迁移 CraftGround 环境`。
- 提交前执行 `git diff --check`，确认无意外生成物、大文件和个人路径。
