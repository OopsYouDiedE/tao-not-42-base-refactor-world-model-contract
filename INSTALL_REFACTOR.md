# 安装系统重构总结

## 📋 改动概览

将项目从**单一 requirements.txt** 重构为 **uv + pyproject.toml + 自动环境检测脚本**。

| 方面 | 之前 | 之后 |
|------|------|------|
| **包管理器** | pip | **uv**（更快的并行依赖解决） |
| **配置文件** | requirements.txt | **pyproject.toml** + requirements/ |
| **环境检测** | 手动 | **自动脚本 (install_env.py)** |
| **Python 版本** | 3.9+ | **>= 3.11**（Colab 标准） |
| **Colab 支持** | 需要手动 apt-get | 自动检测 + 安装虚拟显示 |
| **Godot 支持** | 不支持 | **完全支持**（Mono 自动装） |
| **Craftground** | 不支持 | **新增支持**（Java 21 自动装） |

---

## 📁 新增文件

### 1. **`pyproject.toml`** — 现代化配置
- 统一的项目元数据和依赖定义
- 支持**可选依赖分组** (`[crafter]`, `[ppo-ad]` 等)
- tool 配置：pytest, black, isort, mypy
- uv 配置：Python 版本管理

**关键改动**：
```toml
[project.optional-dependencies]
crafter = ["crafter", "ray[tune]>=2.0"]
ppo-ad = ["crafter", "pot", "scikit-optimize"]
dreamer = ["crafter", "tensorflow>=2.13"]
craftground = ["craftground>=0.1.0"]  # 新增
godot = ["godot-python>=0.5.0"]       # 新增
colab = ["pyvirtualdisplay"]          # 环境特定
```

### 2. **`install_env.py`** — 自动环境检测脚本
- 💡 智能检测运行环境（本机 / Colab / Godot）
- 🔧 自动安装系统依赖（apt-get）
- 🐍 推荐相应的 Python 包组合
- ⚡ 使用 uv 进行高效安装

**支持的命令**：
```bash
python install_env.py                  # 交互式
python install_env.py --colab          # Colab 配置
python install_env.py --godot          # Godot 支持
python install_env.py --craftground    # Craftground
python install_env.py --full           # 全部
```

### 3. **`requirements/` 目录** — 备选传统方式
为了向后兼容和支持 `pip install -r` 用户：
```
requirements/
├── base.txt          # 核心
├── crafter.txt       # Crafter
├── ppo-ad.txt        # PPO+AD
├── dreamer.txt       # DreamerV3
├── craftground.txt   # Craftground（新）
├── minecraft.txt     # Minecraft
├── rl.txt            # RL 工具
├── godot.txt         # Godot（新）
├── dev.txt           # 开发工具
└── all.txt           # 全部
```

### 4. **更新的文件**

- **`INSTALL.md`** — 完整重写，包括 uv、自动脚本、环境特定说明
- **`setup.py`** — 保留向后兼容（但推荐用 pyproject.toml）
- **`requirements.txt`** — 改为指向 INSTALL.md 的说明文档

---

## 🎯 使用方式对比

### 之前（pip）
```bash
pip install -r requirements.txt
```
❌ 装全部，不分模块，容易冲突

### 现在方式 1：自动脚本（最推荐）
```bash
python install_env.py --colab --ppo-ad
```
✅ 自动检测环境、装系统依赖、推荐模块

### 现在方式 2：手动 uv（高级用户）
```bash
uv pip install -e .[ppo-ad,dev]
```
✅ 快速、并行依赖解决、易于组合

### 现在方式 3：传统 pip -r（备选）
```bash
pip install -r requirements/ppo-ad.txt
```
✅ 熟悉的用法，但速度较慢

---

## 🔑 核心改进

### 1️⃣ **模块化依赖**
```bash
# 之前：装全部
pip install -r requirements.txt

# 现在：按需装
pip install -e .[ppo-ad]        # 只装 PPO+AD
pip install -e .[crafter]       # 只装 Crafter
pip install -e .[colab,ppo-ad]  # 组合装
```

### 2️⃣ **自动环境检测**
```bash
# 之前：需要手动输入
!apt-get install -y openjdk-21-jdk ...
!pip install pyvirtualdisplay

# 现在：一行搞定
python install_env.py --colab  # 自动检测 Colab，装虚拟显示
```

### 3️⃣ **快速依赖解决**
```bash
# 使用 uv（5-10倍于 pip）
uv pip install -e .[all]  # 几秒钟
```

### 4️⃣ **新环境支持**

#### Colab
```bash
python install_env.py --colab
# 自动：检测虚拟显示需求 → apt-get install xvfb → pip install pyvirtualdisplay
```

#### Godot
```bash
python install_env.py --godot
# 自动：apt-get install mono-complete → pip install godot-python
```

#### Craftground
```bash
python install_env.py --craftground
# 自动：apt-get install openjdk-21-jdk → pip install craftground
```

---

## 📊 Python 版本支持

| 版本 | 支持 | 说明 |
|------|------|------|
| 3.10 | ❌ | 不再支持 |
| 3.11 | ✅ | 推荐最低版本 |
| 3.12 | ✅ | 完全支持 |
| 3.13 | ✅ | **Colab 默认** |

uv 会自动处理版本差异，无需手动干预。

---

## 🚀 迁移步骤

如果你之前用 `pip install -r requirements.txt`：

### 方式 A：完整迁移（推荐）
```bash
# 1. 删除旧环境
deactivate
rm -rf .venv

# 2. 创建新环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 3. 自动安装
python install_env.py
```

### 方式 B：保守迁移（不改环境）
```bash
# 直接用 setup.py + uv
uv pip install -e .[ppo-ad]
```

### 方式 C：继续用 pip（最小改动）
```bash
pip install -r requirements/ppo-ad.txt
```

---

## 📝 依赖组速查表

| 场景 | 命令 |
|------|------|
| 只要核心库 | `pip install -e .` |
| Crafter 训练 | `pip install -e .[crafter]` |
| PPO+AD | `pip install -e .[ppo-ad]` |
| DreamerV3 | `pip install -e .[dreamer]` |
| Colab + PPO-AD | `pip install -e .[colab,ppo-ad]` |
| Godot RL | `pip install -e .[godot]` |
| Craftground | `pip install -e .[craftground]` |
| 开发工具 | `pip install -e .[dev]` |
| 全部 | `pip install -e .[all]` |

---

## 💡 建议

### 对于 Colab 用户
```python
# 在 Cell 中直接运行
!python install_env.py --colab --ppo-ad
```

### 对于本地开发
```bash
# 安装开发工具
python install_env.py --full
```

### 对于快速原型
```bash
# 仅必需的
uv pip install -e .[crafter]
```

---

## ❓ FAQ

### Q：还能用 `pip install -r requirements.txt` 吗？
A：可以，但不推荐。改用 `pip install -e .` 或 `uv pip install -e .`。

### Q：为什么改成 Python >= 3.11？
A：Colab 默认 3.13，主流版本是 3.11/3.12。避免老旧 Python 的兼容问题。

### Q：uv 和 pip 可以混用吗？
A：可以，但建议一致。uv 更快，推荐优先用 uv。

### Q：我只想用最小化依赖怎么办？
A：`pip install -e .` 只装核心库，其他按需加。

### Q：Colab 中自动装 xvfb 失败了怎么办？
A：运行 `python install_env.py --colab --skip-python-deps`，只装系统依赖，然后手动 `pip install pyvirtualdisplay`。

---

## 🔗 相关文档

- **[INSTALL.md](INSTALL.md)** — 详细安装指南（新）
- **[setup.py](setup.py)** — 向后兼容的项目配置（保留）
- **[pyproject.toml](pyproject.toml)** — 现代化配置（推荐）
- **[readme.md](readme.md)** — 项目说明（已更新）

---

## 📈 预期收益

| 指标 | 提升 |
|------|------|
| **安装速度** | 5-10× 更快（uv 并行） |
| **Colab 友好度** | ⭐⭐⭐⭐⭐（自动虚拟显示） |
| **新手友好度** | ⭐⭐⭐⭐ （自动脚本引导） |
| **依赖管理** | ⭐⭐⭐⭐⭐ （按需模块化） |
| **环境扩展性** | ⭐⭐⭐⭐⭐ （轻易添加新环境） |

---

**现在，用户可以轻松选择自己需要的模块，uv 会快速并行解决依赖，自动脚本智能检测环境并安装必需的系统依赖。** 🎉
