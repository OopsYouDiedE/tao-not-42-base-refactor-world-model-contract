# 代码组织规范(放置 / 写作 / 拆分合并)

> 本文是**硬约束**,对 AI 助手与人类开发者同等生效。它回答三个问题:
> **东西放哪**(放置位置)、**怎么写**(写作格式)、**何时拆何时并**(拆分合并)。
> 与 [AGENTS.md](../AGENTS.md) 互补:AGENTS 管数值不变量(I1–I8)与生产纯净,本文管结构。

---

## 1. 放置位置(目录契约)

判据一句话:**按"是什么"分层,不按"属于哪个实验"分层。** 一段代码该放哪,
只看它的**性质**(算子 / 模型 / 数据契约 / 训练循环 / 工具 / 测试),不看它服务于哪个 run。

| 层 | 目录 | 放什么 | 禁止放 |
|---|---|---|---|
| L1 积木 | `blocks/` | 与任务无关的可复用算子(注意力 / 门控残差 / 时间编码 / SIGReg) | 任何领域字眼(Minecraft/VPT)、训练逻辑 |
| L2 网络 | `net/` | 模型与部件:`world_model.py` 主模型、`slots.py`、`backbone.py`、`heads.py` | 训练循环、loss、mock、数据加载 |
| L2.5 第三方 | `net/vpt_lib/` | 原样 vendored 的 OpenAI VPT(见其 `NOTICE`) | 我们改写的代码;**本规范不约束 vendored 目录** |
| L3 领域契约 | `domains/<game>/` | 数据契约与领域逻辑:动作编解码、数据集、控制重映射、任务文本 | 训练循环、模型定义 |
| L4 训练 | `train/<game>/` | 只放"循环 + 装配":`train_*.py`(CLI/main)、`losses.py`、`eval.py`、`viz`、`_seq.py` | 模型定义、数据契约 |
| 工具 | `tools/` | 一次性脚本 / 离线诊断(oracle、数据下载) | 大体积产物(→ `runs/`)、生产依赖 |
| 测试 | `tests/` | **所有** mock、CPU 兼容、离线降级;`unit/` 与 `integration/` | — |
| 文档 | `knowledge/` | 宏观设计意图与"为什么"(中文,SSOT) | 历史活动流水账(→ git log) |
| 产物 | `runs/` | 数据 / checkpoints / 日志 **[gitignored]** | 任何要入库的源码 |

**依赖方向**(只许向下,禁止回边):
```
blocks ← net ← domains ← train ← tools
                  ↑________________↑   (domains 是 net 与 train 共同的下游契约)
utils 横向供给各层;tests 依赖一切但不被依赖。
```
- `net/` **不得** import `domains/` 或 `train/`(契约数值如 `N_CAMERA_BINS` 在 net 内独立声明,
  由训练端断言一致;见 `net/heads.py`)。
- `train/` 内禁止出现 `train_X ← eval ← train_X` 这类回边;共享的无状态 helper 下沉到
  独立模块(如 `train/minecraft/_seq.py`)打断循环。

---

## 2. 写作格式

1. **文件头 docstring**:一句话职责 + 一行"对外接口清单"(列出本文件导出的类/函数)。
   禁止在文件头写历史 / 复盘 / "本次改了啥"——那进 git commit message。
2. **类 / 函数 docstring**(承 AGENTS §1):NumPy 风格,**必须**声明每个张量参数与返回的
   `Shape` 与 `Dtype`,以及单位("帧 / 秒 / 度 / 像素")。
3. **注释分两类,物理隔离**:
   - **接口契约 / 不变量**(Shape、I1–I8、单位、为何 fp32)→ 留在代码紧邻处。
   - **设计动机 / 复盘"为什么这么改"**(成段的实验叙事)→ 沉到 `knowledge/`,代码里只留
     `# 见 knowledge/xxx.md §N` 一行指针。新增代码按此办;既有的成段注释逐步迁移,不强制一次清。
4. **命名**:标识符英文、docstring/说明中文。**靠目录消歧,不靠前缀**——`net/` 下只有一个
   世界模型时用 `DecoderHeads` 而非 `MinecraftDecoderHeads`。
5. **import**:依赖一律写在**文件顶部**,禁止函数内跨模块借用(隐藏依赖);禁止 `from x import *`
   桶导出(命名空间不透明)——`blocks/__init__.py` 是显式 re-export 的范例。
6. **写作语气**(承 AGENTS §7):禁止情绪化夸张词;结论用数据 / 收敛指标 / 物理原理说明。

---

## 3. 拆分与合并规范

**拆分触发(满足任一即拆)**:
- 文件 > **500 行**,或一个文件里出现 **≥2 类无依赖职责**(如旧 train_minecraft 同时装
  loss + eval + 训练循环 + CLI)。
- `main()` > **80 行**,或函数内嵌套闭包 ≥3 个 → 闭包提升为模块级函数。
- 同一概念出现**第 2 份实现**(曾经两份 `StateDecoder` / 两个 `sinusoidal_time_encoding`)
  → 立即合并到单一定义,放进它该在的层。

**合并触发(满足即并)**:
- 一个文件 < ~40 行、只被单一模块用、又无独立测试 → 并入调用方。
- 跨文件重复的小工具 → 收敛到最低共同层(算子去 `blocks/`,契约常量去 `domains/`)。

**拆分时保持无环**:抽出的子模块只能依赖比它更低的层。若两个同层模块都要用某 helper,
helper 下沉到二者的公共下游(参见 `_seq.py` 打断 `train ↔ eval` 循环的做法)。

---

## 4. 已知技术债(待清,不阻塞)

- `utils/__init__.py` 仍用 `from .x import *` 桶导出(无人依赖 `from utils import *`,危害低);
  按 §2.5 应改显式 re-export 或删桶。
- `net/world_model.py`、`domains/` 等文件头仍保留成段设计叙事(历史遗留);按 §2.3 应逐步下沉
  到 `knowledge/`,代码留指针。
- `knowledge/mental_world.md` 描述的是**设计愿景**(含 fovea/gaze 等未落地于当前活模型的部件),
  与 `net/world_model.py` 的当前实现是"愿景 vs 现状"关系,已在该文顶部标注。
