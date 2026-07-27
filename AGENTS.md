# 项目开发规范

约束本仓库的代码、测试、文档与自动化修改，人类与 AI 助手同样适用。

## 1. 当前范围

仓库只维护两条生产路径：

- `bc_datasets/minestudio/`：MineStudio 数据集批量下载 + 转 Lumine 格式预训练数据；
- `train/`：Unsloth 视觉 SFT，Gemma 4 与 Qwen3.6 两族主干。

不在范围内：在线环境（Godot / CraftGround / solaris）、控制契约、mineflayer 采集、
世界模型、RL / GRPO、IDM、VPT 教师、判官。新增这些能力必须由用户明确要求，
不得根据历史文件名自行恢复。

## 2. 目录职责

| 目录 | 职责 |
|---|---|
| `bc_datasets/` | 行为克隆数据集构建的命名空间；每个子包一个数据来源 |
| `bc_datasets/minestudio/` | 数据下载、LMDB 读取、Lumine 动作编解码、预训练数据构建 |
| `machine_environment/` | 本机 CPU / 内存 / 磁盘 / GPU / CUDA 检测 |
| `train/` | Unsloth 视觉 SFT 流程与两族主干入口 |
| `tests/unit/` | 不碰真实数据集与模型的纯单元测试 |
| `runs/bc_datasets/` | 数据集：原始 LMDB 分片与 Lumine 转换产物 |
| `runs/trains/` | 训练产物：checkpoint、LoRA adapter、训练日志 |

依赖方向单向：`bc_datasets ← train`。`bc_datasets/` 不得 import `train/`。
新增数据来源作为 `bc_datasets/` 下的兄弟子包，不得塞进 `minestudio/`。

## 3. 命名规范

- 文件、目录、模块、公开类型与函数用完整描述性英文单词。
- Python 用 `snake_case`。
- VPT、LMDB、RGB、LoRA、SFT、MoE、API 等行业标准缩写可保留。
- 禁止项目私有缩写，例如新写 `env`、`cfg`、`img`、`act` 作为公开名称。
- 张量公式里的 `B/T/H/W/C` 是数学符号，可保留。
- 顶层包不得命名为 `datasets`：`import unsloth` 内部 `from datasets import Dataset`，
  会被同名顶层包遮蔽导致 unsloth_zoo 崩（`python -m` 从仓库根跑必现）。
  `bc_datasets` 带前缀，不构成遮蔽，可以用。

## 4. 依赖规则

- 依赖必须由生产代码中的实际 import 证明，统一声明在 `pyproject.toml`。
- 训练侧依赖（torch / transformers / trl / unsloth）放在 `train` extra，
  数据管线机器不必装 CUDA 栈。
- 缺少生产依赖时直接报告，不在生产代码里加 `try/except` Mock 或静默降级。
  `machine_environment` 对 torch 的 `try/except ImportError` 是例外：它的职责就是
  报告环境缺什么，且必须能在没装 CUDA 栈的数据管线机器上运行。缺失结果显式标注，
  不填默认值。
- `import unsloth` 必须早于 transformers / trl 的重型导入，这是补丁顺序要求。

## 5. 数据契约不变量

- MineStudio LMDB 的 key 是 `str((episode_idx, 起始帧号))`，第二项是**帧号**
  （`chunk_size` 的整数倍），不是块序号。
- 跨模态对齐走 episode 名，不得按分片号配对：各模态分片切分边界不同。
- LMDB 不允许同进程重复打开同一环境。已持有 `ModalKernelReader` 时要划分数据，
  必须把帧数传给 `build_split(episode_frames=...)`，不能让它再开一次。
- `ModalKernelReader` 返回的数组必须与内部块缓存解耦：单块命中时切片是缓存对象的
  视图，必须 copy 后返回，否则调用方原地修改会污染缓存。
- episode 名中间的 12 位 hex 不是可靠的会话标识：10xx 中 `f153ac423f61` 横跨全部
  19 个前缀。分组只用前缀或整条 episode。
- 动作解码永远返回定长、结构合法的结果：未知键名丢弃，非法数值按 0，
  chunk 数不足补空、超出截断。大模型吐出的文本再脏也不能让解码抛错。
- 鼠标增量钳到 ±999，滚轮钳到 ±5。
- 相机换算固定 0.15 度/像素（VPT 口径）；分母必须挡住非正输入。

## 6. 训练边界

- 离线 loss 不是闭环能力结论。闭环结论必须报告固定 seed、样本量与成功指标。
- 报告验证集指标必须同时说明 holdout 粒度。`episode` 粒度的数字不能当作跨玩家
  泛化证据——同一玩家在两边都有。
- checkpoint 结构不兼容时显式升级名称或版本，禁止静默部分加载。
- chat template 与 EOS token 在训练和推理端必须一致——不一致是导出后
  效果变差的最常见原因。

## 7. 代码与文档

- 项目 Markdown 用中文，标识符用英文。
- 文件头 docstring 写一句职责和对外接口，不记录修改历史。
- 类和函数 docstring 用 NumPy 风格；张量参数与返回值声明 Shape、Dtype 与单位。
- import 位于文件顶部，禁止 `from module import *`（unsloth 的延迟导入需注明原因）。
- 生产代码不得包含测试 Mock、离线假数据或本地个人绝对路径。
- `runs/` 只有 `bc_datasets/` 与 `trains/` 两个子目录，整体 Git ignored。
  临时脚本、探针与一次性产物不留在 `runs/`：用完即删，需要留存的结论写进代码或文档。
- 废弃模块物理删除，不留空文件、转发壳或仅含注释的兼容层。

## 8. 测试与验收

    python -m pytest
    python -m compileall -q bc_datasets train tests

单元测试不得依赖已下载的数据集或模型权重。涉及真实 LMDB、CUDA 或模型加载的验证
属于冒烟，需单独说明在哪台机器上跑过；纯 CPU 测试不能替代。

测试失败时必须说明失败命令与直接原因。不得把"缺少本地依赖"描述为代码测试通过。

## 9. Git 与修改纪律

- 保留用户已有的未提交修改；不得用 `git reset --hard` 或覆盖式 checkout 清理工作树。
- 删除、移动与批量重命名前先核对精确目标，并同步所有引用。
- 每轮完成的修改创建一个职责单一的 commit，message 用中文说明结果。
- 提交前执行 `git diff --check`，确认无意外生成物、大文件与个人路径。
