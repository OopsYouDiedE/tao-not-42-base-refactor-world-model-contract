# 安装指南 — uv + 模块化可选依赖

本项目使用 **uv** 管理依赖，支持 **Python >= 3.11**，并支持**按需安装**模块。

## 🚀 最简快速开始

### 方法 1：自动配置（推荐）
脚本**自动检测平台**（Colab / 本机 / 服务器），按需添加系统依赖。

```bash
# 交互式问答
python install_env.py

# 直接指定模块（脚本自动适配平台）
python install_env.py --ppo-ad              # PPO+AD（自动装虚拟显示，如果在 Colab）
python install_env.py --dreamer             # DreamerV3
python install_env.py --ppo-ad --dev        # 组合多个模块
python install_env.py --full                # 全部
```

**核心特点**：
- ✅ 自动检测 Colab / 本机 / 服务器环境
- ✅ 自动安装系统依赖（apt-get）
- ✅ 自动添加平台特定的 Python 包（如虚拟显示）
- ✅ 用户只需指定**功能模块**，平台差异由脚本处理

### 方法 2：手动安装（高级用户）
```bash
# 仅核心依赖
pip install -e .

# 指定模块
uv pip install -e .[ppo-ad]           # 快速，使用 uv
pip install -e .[ppo-ad]              # 传统方式

# 组合多个模块
uv pip install -e .[ppo-ad,dev]
uv pip install -e .[crafter,ppo-ad,dreamer]  # 完整 Crafter
```

---

## 关于 uv

**uv** 是一个极快的 Python 包管理器，具有以下优势：

- ⚡ **并行下载**：比 pip 快 5-10 倍
- 🔧 **自动依赖解决**：智能处理版本冲突
- 📦 **Python 版本管理**：自动下载正确的 Python 版本（如果需要）
- ✨ **Colab 友好**：完美支持 Python 3.13

### 安装 uv
```bash
pip install uv

# 或在 Colab：
!pip install uv
```

### 使用 uv 代替 pip
```bash
# 替代 pip install
uv pip install numpy

# 替代 pip install -e
uv pip install -e .

# 创建虚拟环境
uv venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate      # Windows
```

---

## 可选模块

### 🎮 **Crafter 训练** — `[crafter]`
用于在 Crafter 环境中训练任何算法。

```bash
# 使用 uv（推荐）
uv pip install -e .[crafter]

# 或使用 pip
pip install -e .[crafter]
```

**包含**：
- `crafter` — 游戏环境
- `ray[tune]` — 分布式训练（可选，但推荐用于子进程并行）

**适用于**：
- `python -m train.crafter.train_ppo_ad`
- `python -m train.crafter.train_dreamerv3`

---

### 🎯 **PPO + Achievement Distillation** — `[ppo-ad]`
专门用于 AD 算法训练。

```bash
pip install -e .[ppo-ad]
```

**包含**：
- `crafter`
- `pot` — Python Optimal Transport（最优传输匹配）
- `scikit-optimize` — 优化工具

**适用于**：
```bash
python -m train.crafter.train_ppo_ad \
  --vec subproc --n-envs 16 --total-timesteps 3000000
```

---

### 🌍 **DreamerV3 世界模型** — `[dreamer]`
用于 DreamerV3 训练和评估。

```bash
pip install -e .[dreamer]
```

**包含**：
- `crafter`
- `tensorflow>=2.13` — 仅用于比对基线（可选）

**适用于**：
```bash
python -m train.crafter.train_dreamerv3
```

---

### ⛏️ **Minecraft 数据处理** — `[minecraft]`
用于处理 Minecraft VPT 数据集。

```bash
pip install -e .[minecraft]
```

**包含**：
- `minerl` — Minecraft 环境接口
- `pillow` — 图像处理

**适用于**：
- `train/minecraft/vpt_dataset.py` — VPT 数据集解码

---

### ⚒️ **Craftground 游戏环境** — `[craftground]`
Craftground 是一个基于 Java 的游戏环境（类似 Minecraft 的简化版）。

```bash
uv pip install -e .[craftground]
```

**包含**：
- `craftground>=0.1.0` — Python 包
- **系统依赖**：Java 21（自动安装）

**自动安装**：
```bash
python install_env.py --craftground
```

---

### 🤖 **RL 工具集** — `[rl]`
强化学习基础工具。

```bash
uv pip install -e .[rl]
```

**包含**：
- `gymnasium>=0.29` — 环保的 gym 替代
- `envpool` — 高速向量环境

---

### 🎮 **Godot RL 环境** — `[godot]`
Godot 引擎 + C# RL 环境支持。

```bash
uv pip install -e .[godot]
```

**包含**：
- `godot-python>=0.5.0` — Godot Python 绑定
- **系统依赖**：Mono（自动安装）

**自动安装**：
```bash
python install_env.py --godot
```

**注意**：需要 Godot 编辑器或 Godot C# 运行时环境。

---

### 🛠️ **开发工具** — `[dev]`
用于代码开发、测试、格式化。

```bash
uv pip install -e .[dev]
```

**包含**：
- `pytest` — 单元测试
- `black` — 代码格式化
- `isort` — import 排序
- `flake8` — 代码检查
- `mypy` — 类型检查

**使用**：
```bash
pytest tests/                    # 运行单元测试
black train/ net/ blocks/       # 格式化代码
isort train/ net/ blocks/       # 整理 import
```

---

---

## 环境特定安装

### 💻 **Google Colab 环境**

Colab 默认运行 Python 3.13。需要虚拟显示（xvfb）来渲染环境。

```bash
# 自动配置（推荐）
python install_env.py --colab --ppo-ad

# 或手动安装
uv pip install -e .[colab,ppo-ad]
```

这会自动安装：
- 虚�virtual 显示依赖（xvfb, pyvirtualdisplay）
- OpenGL 库（libgl1-mesa-dev 等）
- Crafter + PPO+AD

**在 Colab Cell 中直接运行**：
```python
!python install_env.py --colab --ppo-ad
```

### 🎮 **Godot C# RL 环境**

如果需要在 Godot 引擎中进行 RL 实验：

```bash
# 自动配置（包括 Mono 安装）
python install_env.py --godot

# 或手动
uv pip install -e .[godot]
```

这会安装：
- Godot Python 绑定
- Mono（C# 运行时）
- RL 基础工具

### ⛏️ **Craftground 环境**

Craftground 需要 Java 21：

```bash
# 自动配置（包括 Java 21）
python install_env.py --craftground

# 或手动
uv pip install -e .[craftground]
```

---

## 组合安装

可以同时安装多个模块，用逗号分隔：

```bash
# Colab 标准配置（Crafter + PPO-AD + Colab 虚拟显示）
uv pip install -e .[colab,ppo-ad]

# 安装 Crafter + PPO-AD + 开发工具
uv pip install -e .[crafter,ppo-ad,dev]

# 安装 Crafter + DreamerV3 + 开发工具
uv pip install -e .[crafter,dreamer,dev]

# Godot + RL 完整配置
uv pip install -e .[godot-full,dev]

# 安装全部（包括所有可选）
uv pip install -e .[all]
```

或者用预定义的组合组（见 `pyproject.toml`）：
```bash
uv pip install -e .[crafter-full]      # Crafter 完整版
uv pip install -e .[colab-ppo-ad]      # Colab 标准版
uv pip install -e .[godot-full]        # Godot 完整版
```

---

## 完整安装场景

### 场景 1：只想跑 PPO+AD（本机）
```bash
python install_env.py --crafter --ppo-ad

# 或手动
uv pip install -e .[ppo-ad]
python -m train.crafter.train_ppo_ad --n-envs 16
```

### 场景 2：Colab 环境跑 PPO+AD
```bash
# 在 Colab Cell 中运行
!python install_env.py --colab --ppo-ad

# 然后
!python -m train.crafter.train_ppo_ad --n-envs 8
```

### 场景 3：想比较 Dreamer 和 PPO-AD
```bash
python install_env.py --crafter --ppo-ad --dreamer

# 或手动
uv pip install -e .[crafter-full]
python -m train.crafter.train_dreamerv3
python -m train.crafter.train_ppo_ad
```

### 场景 4：Godot RL 开发
```bash
python install_env.py --godot --dev

# 或手动
uv pip install -e .[godot,dev]
# 开启 Godot 编辑器进行 RL 实验...
```

### 场景 5：Craftground 环境
```bash
python install_env.py --craftground

# 或手动
uv pip install -e .[craftground]
# Craftground 环境已就绪，可开始训练
```

### 场景 6：开发新模块（完整环境）
```bash
python install_env.py --full

# 或手动
uv pip install -e .[all]

# 修改代码 → 测试 → 提交
pytest tests/
black train/ && isort train/
mypy train/
```

### 场景 7：最小化 Colab 环境
```bash
# 只装必需的（Crafter + 虚拟显示）
!python install_env.py --colab
```

---

## 故障排除

### Python 版本不满足
**报错**：`Python X.Y 不满足（需要 >= 3.11）`

**解决**：升级到 Python 3.11+。在 Colab 中默认已满足。

```bash
# 查看当前版本
python --version

# 如果需要升级，在本机：
# macOS: brew install python@3.11
# Ubuntu: sudo apt-get install python3.11
# Windows: 从 python.org 下载
```

### 导入报错：`ModuleNotFoundError`
**原因**：没有装对应的可选模块。

**解决**：运行自动安装脚本，或手动安装缺失的模块。

```bash
# 如果缺 crafter
python install_env.py --crafter

# 如果缺 tensorflow（DreamerV3）
python install_env.py --dreamer

# 或手动
uv pip install -e .[ppo-ad]
```

### 系统依赖缺失
**Colab 虚拟显示报错**：`xvfb not found` 或 `pyvirtualdisplay`

**解决**：
```bash
# 自动安装
python install_env.py --colab

# 或手动
!apt-get update && apt-get install -y xvfb
!pip install pyvirtualdisplay
```

**Godot 相关报错**：缺 Mono

```bash
python install_env.py --godot
```

### 某个库版本冲突
**原因**：不同的依赖组要求不同的库版本。

**解决**：使用 uv（会自动解决），或明确指定版本：

```bash
# uv 会自动找到兼容的版本
uv pip install -e .[crafter,minecraft]

# 如果用 pip，可能需要手动指定
pip install torch==2.1.2 torchvision==0.16.2
```

### 完全重新安装
```bash
# 删除虚拟环境
rm -rf .venv

# 重新创建
python -m venv .venv
source .venv/bin/activate

# 重新安装
python install_env.py --full
```

### 在 Colab 中重新安装
```python
# 清空并重装
!pip uninstall -y tao-not-42
!python install_env.py --colab --ppo-ad
```

---

## Python 版本和 uv 兼容性

| Python 版本 | 支持情况 | 推荐环境 |
|------------|--------|--------|
| 3.10 | ❌ 不支持 | — |
| 3.11 | ✅ 完全支持 | 本机 / 服务器 |
| 3.12 | ✅ 完全支持 | 本机 / 服务器 |
| 3.13 | ✅ 完全支持 | **Colab 默认** |

**uv 能自动处理版本差异**，无需手动干预。

---

## 验证安装

安装完成后，验证主要模块：

```bash
# 核心依赖
python -c "import torch; print(f'✅ torch {torch.__version__}')"

# Crafter（如果装了）
python -c "import crafter; print('✅ crafter OK')" 2>/dev/null || echo "❌ crafter 未装"

# PPO+AD（如果装了）
python -c "import pot; print('✅ POT OK')" 2>/dev/null || echo "❌ POT 未装（需要 [ppo-ad]）"

# DreamerV3（如果装了）
python -c "import tensorflow; print('✅ TensorFlow OK')" 2>/dev/null || echo "❌ TensorFlow 未装（需要 [dreamer]）"

# 虚拟显示（Colab）
python -c "import pyvirtualdisplay; print('✅ pyvirtualdisplay OK')" 2>/dev/null || echo "❌ 虚拟显示未装（需要 [colab]）"
```

---

## 依赖树（pyproject.toml）

```
核心依赖（所有用户）
├─ numpy
├─ torch >= 2.0
├─ torchvision
├─ opencv-python
├─ transformers >= 4.30
├─ matplotlib
├─ wandb / requests / pyyaml
│
可选模块分组：
├─ [crafter]
│  ├─ crafter
│  └─ ray[tune] >= 2.0
│
├─ [ppo-ad]
│  ├─ crafter
│  ├─ pot
│  └─ scikit-optimize
│
├─ [dreamer]
│  ├─ crafter
│  └─ tensorflow >= 2.13
│
├─ [craftground]
│  ├─ craftground >= 0.1.0
│  └─ [系统] Java 21
│
├─ [minecraft]
│  ├─ minerl
│  └─ pillow >= 9.0
│
├─ [rl]
│  ├─ gymnasium >= 0.29
│  └─ envpool
│
├─ [godot]
│  ├─ godot-python >= 0.5.0
│  └─ [系统] Mono
│
└─ [dev]
   ├─ pytest / pytest-cov
   ├─ black / isort
   └─ flake8 / mypy
```

---

## 环境变量（可选）

某些库需要额外配置：

```bash
# CUDA 12.4 用户（见 memory）
export CUDA_HOME=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# HuggingFace 下载（如果使用代理）
pip install socksio
export HF_ENDPOINT=https://huggingface.co

# Godot 环境
export GODOT_BIN=/path/to/godot4  # 如果需要指定 Godot 路径
```

---

## 下一步

安装完成后，根据你的场景：

### 🎯 快速开始 PPO+AD
```bash
python -m train.crafter.train_ppo_ad \
  --n-envs 16 \
  --total-timesteps 3000000 \
  --run-dir runs/crafter_ppo_ad
```

### 🌍 运行 DreamerV3
```bash
python -m train.crafter.train_dreamerv3
```

### 🔬 进行开发 / 调试
```bash
# 运行测试
pytest tests/

# 格式化代码
black train/ net/ blocks/ && isort train/ net/ blocks/

# 类型检查
mypy train/ --ignore-missing-imports
```

### 📖 查看文档
- [知识库](knowledge/) — 设计文档（mental_world.md, dreamer.md, ppo_ad.md）
- [项目结构](readme.md#项目结构) — 代码组织说明
- [代码规范](AGENTS.md) — 贡献指南

---

**有问题？** 检查 [故障排除](#故障排除) 部分，或创建 Issue。
