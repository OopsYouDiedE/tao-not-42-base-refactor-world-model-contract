# 助手约束与开发规范 (AGENTS.md)

> **文档性质**：对 AI 助手和人类开发者均有约束力的规则文档。只记录**规则**，不记录历史活动（历史活动归 git log）。
> §1–§7 是数值不变量与生产纪律，§8–§11 是代码组织规范（放置 / 写作 / 拆分合并），§12 是 SubAgent 使用规范。
> Godot 子系统有自己的局部规范，见 [assets/godot_meta_rl/AGENTS.md](assets/godot_meta_rl/AGENTS.md)。

---

## 1. 文档规范

- 所有项目文档（`.md` 文件）必须使用**中文**撰写和更新。
- **SSOT 原则**：代码是逻辑的唯一事实来源。Markdown 只说明宏观架构、算法设计思路与物理含义，禁止在 Markdown 中描述容易变动的微观实现细节。
- 所有 Class / Function 必须用 NumPy / Google 风格 Docstring 明确声明 Shape 和 Dtype 契约。
- 任何对 `net/` 或 `train/` 的目录结构、类名、核心方法签名的修改，**必须同步更新** `knowledge/` 中对应专题与 `knowledge/code_analysis.md`。
- 废弃模块必须**物理删除**原文件，禁止留空文件或仅写注释，并在 commit message 中明确描述迁移路径。
- 每次完成一轮修改后必须提交一次 Git commit。

---

## 2. 生产代码纯净原则

- 核心代码（`net/`、`train/`）必须保持干净，**禁止**混入任何本地 Mock、调试数据加载或环境兼容降级逻辑。
- 所有依赖（`transformers`、`opencv-python`、`datasets` 等）**缺包直接报错**，不写 `try/except` 降级。
- CPU 兼容、骨干 Mock、离线数据 Mock **必须且仅允许**存在于 `tests/` 目录。骨干 mock 通过
  **依赖注入**传入，生产 `net/` 不包含任何 mock。
- 模型结构参数（d/N/K/J、骨干 / 各部件的选择与超参）通过 `configs/<game>/*.yaml` 预设配置传入。
  `net/` 只持有类型化 schema（`net/config.py`，纯 dataclass 无 IO）与 `build_*` 工厂；yaml 读取在
  `utils/io.py`（`load_yaml`）、领域常量校验在 `train/`——`net/` 不 import domain、不读文件。

---

## 3. 测试目录结构

```
tests/
  unit/                纯单元测试（无需网络/GPU：SIGReg、空间位置编码）
  integration/         集成测试（DI 注入 mock 骨干，离线 CPU 前向+反向；如 test_dreamer_build）
```

所有 mock（骨干 mock、离线数据）按 §2 放在 `tests/`，骨干 mock 经依赖注入传入模型。

---

## 4. 环境矩阵

| 硬件环境 | 可运行内容 |
| :--- | :--- |
| **Linux + CUDA（生产）** | 全量：net/ + train/（含各数据集领域契约）+ 真实 DINOv3 骨干训练 |
| **Windows + CUDA（开发）** | net/ 前向、`tests/`（DI mock 骨干离线冒烟；Mamba 已弃用，无平台门槛） |
| **CPU Only** | `tests/unit/` + `tests/integration/`（依赖注入 mock 骨干，小尺寸） |

> 视觉骨干统一走 HuggingFace `transformers`（见 `net/backbone.py`）。DINOv3 权重受访问限制，
> 需 HF token（`utils/io.py` 的 `get_hf_token`）；无 token 可用开放权重 dinov2 预设。

---

## 5. 训练数据与文件系统

- 所有下载数据、checkpoints、训练日志统一存放在 `runs/`（已加入 `.gitignore`，**不入库**）。
- `runs/data/`        下载的数据集缓存
- `runs/checkpoints/` 模型权重
- `runs/logs/`        训练日志

---

## 6. 数值不变量（I1–I8，硬约束）

| # | 不变量 |
|---|---|
| I1 | 除法分母 `clamp(min=ε)`，ε ≥ 1e-4（不用 1e-12） |
| I2 | 不做无界 exp；softmax/Sinkhorn 走 log 域 |
| I3 | 所有头输出构造上有界（tanh/sigmoid/softplus+ε/exp-clamp） |
| I4 | 危险算子（除/normalize/exp/求逆/投影）强制 fp32 |
| I5 | 每个递归/残差更新增益受限（< 1 或 clamp） |
| I6 | 不稳定组合优化（Sinkhorn/匈牙利）只在损失里，不进前向 |
| I7 | 递归/rollout 路径用 LayerNorm/GroupNorm，不用 BatchNorm |
| I8 | 长链梯度受控；长程依赖不取消 detach，改用 stop-grad 对比损失 |

---

## 7. 写作规范

- 禁止使用情绪化或夸张的修饰词（如"暴降"、"完美"、"惊人"、"瞬间"等）。
- 所有结论和成效必须以数据、收敛指标或物理原理为依据。

---

## 8. 放置位置（目录契约）

判据一句话：**按"是什么"分层，不按"属于哪个实验"分层。** 一段代码该放哪，
只看它的**性质**（算子 / 模型 / 训练域[数据契约+循环] / 测试），不看它服务于哪个 run。

| 层 | 目录 | 放什么 | 禁止放 |
|---|---|---|---|
| L1 积木 | `blocks/` | 与任务无关的可复用算子（注意力 / 门控残差 / 时间编码 / 卷积编解码 / 分布 / SIGReg） | 任何领域字眼（Minecraft/Crafter/VPT）、训练逻辑 |
| L2 网络 | `net/` | 模型与部件：`dreamerv3/`、`dreamer4/`、`ppo_ad/` 等世界模型/策略，`backbone.py` 骨干加载；结构 schema `config.py`（纯 dataclass）与各部件 `build_*` 工厂 | 训练循环、loss、mock、数据加载、yaml/文件 IO |
| L2.5 第三方 | `net/vpt_lib/` | 原样 vendored 的 OpenAI VPT（见其 `NOTICE`）。**本规范不约束 vendored 目录**；升级方式是重拉上游覆盖 | 我们改写的代码 |
| L3 训练域 | `train/<game>/` | 该数据集/游戏的**全部领域逻辑 + 训练**：数据契约（动作编解码、数据集、任务文本）、回放、loss、循环/装配 `train_*.py`（CLI/main）。**不同数据集的区分全压在这一层**（`train/crafter/`、`train/minecraft/`、`train/craftground/`、`train/godot_meta_rl/` 各自自洽） | 模型定义、跨域可复用算子 |
| 配置 | `configs/<game>/` | 模型结构 yaml 预设（部件选择 + 超参；缺键取 `net.config` 默认） | 模型定义、训练逻辑、数据契约 |
| 测试 / 离线脚本 | `tests/` | **所有** mock、CPU 兼容、离线降级、骨干冒烟、一次性诊断；`unit/` 与 `integration/` | 生产依赖、大体积产物（→ `runs/`） |
| 文档 | `knowledge/` | 宏观设计意图与"为什么"（中文，SSOT） | 历史活动流水账（→ git log） |
| 产物 | `runs/` | 数据 / checkpoints / 日志 **[gitignored]** | 任何要入库的源码 |

**依赖方向**（只许向下，禁止回边）：
```
blocks ← net ← train
utils 横向供给各层；tests 依赖一切但不被依赖（离线脚本 / mock / 诊断都落在此层）。
```
- `net/` **不得** import `train/`（领域常量在 net 内独立声明，由训练端断言一致）。**亦不读 yaml/文件**：
  结构 schema 是纯 dataclass（`net/config.py`），`configs/<game>/*.yaml` 由 `train` 装配时经 `utils.io` 解析成
  配置对象再传入模型；领域常量（act_dim / n_bins 等）由 `train/<game>/` 持有并断言一致。
- `train/` 内禁止出现 `train_X ← eval ← train_X` 这类回边；共享的无状态 helper 下沉到独立模块打断循环。

---

## 9. 写作格式

1. **文件头 docstring**：一句话职责 + 一行"对外接口清单"（列出本文件导出的类/函数）。
   禁止在文件头写历史 / 复盘 / "本次改了啥"——那进 git commit message。
2. **类 / 函数 docstring**（承 §1）：NumPy 风格，**必须**声明每个张量参数与返回的
   `Shape` 与 `Dtype`，以及单位（"帧 / 秒 / 度 / 像素"）。
3. **注释分两类，物理隔离**：
   - **接口契约 / 不变量**（Shape、I1–I8、单位、为何 fp32）→ 留在代码紧邻处。
   - **设计动机 / "为什么这么改"**（成段的实验说明）→ 沉到 `knowledge/`，代码里只留
     `# 见 knowledge/xxx.md §N` 一行指针。新增代码按此办；既有的成段注释逐步迁移，不强制一次清。
4. **命名**：标识符英文、docstring/说明中文。**靠目录消歧，不靠前缀**——`net/` 下同类只有一个时用
   通用名而非领域前缀名。
5. **import**：依赖一律写在**文件顶部**，禁止函数内跨模块借用（隐藏依赖）；禁止 `from x import *`
   桶导出——`blocks/__init__.py` 是显式 re-export 的范例。
6. **写作语气**（承 §7）：禁止情绪化夸张词；结论用数据 / 收敛指标 / 物理原理说明。

---

## 10. 拆分与合并规范

**拆分触发（满足任一即拆）**：
- 文件 > **500 行**，或一个文件里出现 **≥2 类无依赖职责**（如同时装 loss + eval + 训练循环 + CLI）。
- `main()` > **80 行**，或函数内嵌套闭包 ≥3 个 → 闭包提升为模块级函数。
- 同一概念出现**第 2 份实现** → 立即合并到单一定义，放进它该在的层。

**合并触发（满足即并）**：
- 一个文件 < ~40 行、只被单一模块用、又无独立测试 → 并入调用方。
- 跨文件重复的小工具 → 收敛到最低共同层（算子去 `blocks/`，契约常量去 `train/<game>/`）。

**拆分时保持无环**：抽出的子模块只能依赖比它更低的层。若两个同层模块都要用某 helper，
helper 下沉到二者的公共下游。

---

## 11. 安装与环境约束

### 11.1 安装系统设计原则

核心原则：**用户指定功能模块，脚本自动适配平台**。

```
用户层面：python install_env.py --ppo-ad --dev
                                    ↓
脚本检测层：is_colab() / is_headless() / is_local()
                                    ↓
平台适配层：自动选择 sys_deps（apt-get）和 py_extras
                                    ↓
安装层：uv pip install -e .[py_extras] + apt-get install [sys_deps]
```

**禁止暴露平台前缀**：
- ❌ 不要 `--colab`, `--godot` 这样的平台参数
- ✅ 只有功能参数：`--ppo-ad`, `--dreamer`, `--godot`（Godot 是功能，不是平台）

### 11.2 新增模块的扩展方式

**添加新游戏环境 / 算法**（如 Minecraft RL）：

1. 在 `pyproject.toml` 的 `[project.optional-dependencies]` 添加分组
   ```toml
   minecraft-rl = ["minerl", "specific-lib"]
   ```

2. 在 `install_env.py` 的 `resolve_extras()` 添加对应逻辑
   ```python
   if "minecraft-rl" in modules:
       py_extras.add("minecraft-rl")
       print("✅ 将安装 Minecraft RL")
   ```

3. 脚本自动处理平台：若在 Colab，自动加虚拟显示；若在 Godot 服务器，自动加 Mono

**添加新平台支持**（如 Docker / 云 IDE）：

1. 在 `install_env.py` 的顶部添加检测函数
   ```python
   def is_docker() -> bool:
       return os.path.exists("/.dockerenv")
   ```

2. 在 `resolve_extras()` 中根据平台添加系统/Python 依赖
   ```python
   if platform == "docker":
       sys_deps.add("...")
   ```

### 11.3 依赖分组指导

| 分组 | 类别 | 包含内容 | 备注 |
|------|------|--------|------|
| `crafter` | 环境 | Crafter 游戏引擎 + ray 并行 | 被 ppo-ad / dreamer 依赖 |
| `ppo-ad` | 算法 | crafter + 最优传输 + 优化工具 | 游戏探索 |
| `dreamer` | 算法 | crafter + 深度学习框架 | 世界模型 |
| `craftground` | 环境 | Craftground Java 游戏 | 需要系统依赖：Java 21 |
| `minecraft` | 数据 | Minecraft VPT 数据集处理 | 异步独立使用 |
| `godot` | 环境 | Godot RL 绑定（待确认包名） | 需要系统依赖：Mono |
| `rl` | 工具 | gymnasium + envpool | 通用 RL 基础 |
| `headless` | 环境适配 | pyvirtualdisplay | 自动在 Colab / 无显示服务器添加 |
| `dev` | 开发 | pytest / black / mypy / isort | 仅开发用 |

### 11.4 已知不确定的依赖与已确认依赖

- **`godot-python`**：GitHub 存在但 PyPI 包名和版本需确认；Godot 4 推荐用 C# + Mono，Python binding 可选
- **`craftground`（已确认）**：已验证在 Python 3.13 虚拟环境下直接通过 `pip` 安装可用，真实包名即为 `craftground`。

---

## 12. SubAgent 使用规范（AI 助手）

- **能并行就并行**：相互独立的调查 / 分析 / 检索任务，必须在同一轮里并行启动多个 SubAgent，
  禁止串行排队；存在依赖关系的任务才按序执行。
- **模型匹配任务难度**：
  | 任务性质 | 模型选择 |
  |---|---|
  | 纯检索 / 文件定位 / 枚举清点 | `haiku` |
  | 常规代码阅读 / 单模块总结 | `sonnet` |
  | 架构分析 / 设计权衡 / 跨模块综合 | 继承主会话模型（不降级） |
- SubAgent prompt 必须自带完整上下文（背景、目录范围、输出格式），不依赖子代理自行摸索；
  产出要求带 `file:line` 引用。

---

## 13. 已知技术债（待清，不阻塞）

- `train/minecraft/vpt_dataset.py` 等文件头若仍保留成段设计叙事，按 §9.3 应逐步下沉到 `knowledge/`，代码留指针。
- `knowledge/mental_world.md` 描述的部分内容（fovea/gaze 选择性读取等）是**尚未落地的设计愿景**，与当前
  代码是"愿景 vs 现状"关系，已在该文顶部标注。
