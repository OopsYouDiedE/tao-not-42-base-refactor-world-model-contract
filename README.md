# TAO

## 快速安装

除 CraftGround 外全部直接安装最新兼容版本，不锁定版本号。
默认运行与验证环境是带 GPU 的 Linux；Windows 上涉及真实 CraftGround 的步骤必须在 WSL 2 中执行。

### 1. 系统依赖

纯逻辑开发和静态检查只需要 Python 和 uv：

```bash
pip install uv
```

真实 CraftGround 需要 JDK、CMake、Ninja、JNI 和 OpenGL 开发包（非 macOS 还需 GLEW）。
Ubuntu / WSL 2 Ubuntu：

```bash
sudo apt-get update
sudo apt-get install -y openjdk-21-jdk cmake ninja-build \
  libglew-dev libgl1-mesa-dev libglu1-mesa-dev libglfw3-dev xorg-dev \
  mesa-utils xserver-xorg-core x11-xserver-utils pciutils
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
```

CUDA 渲染与 GPU 训练额外需要 NVIDIA 驱动和 CUDA Toolkit，`nvidia-smi` 必须可用。
无桌面服务器还要自建 NVIDIA Xorg 并设置 `DISPLAY`，`glxinfo -B` 必须显示
`direct rendering: Yes` 与真实 NVIDIA renderer；出现 `llvmpipe` 表示软件渲染，不算通过。

Godot 环境需要 Godot .NET 版与匹配的 .NET SDK，可执行文件路径通过 `GODOT_EXE` 指定
（默认取 `PATH` 中的 `godot`）。

### 2. Python 环境

```bash
uv venv --python 3.13
source .venv/bin/activate    # Windows: .venv\Scripts\activate
```

CPU 或纯逻辑开发：

```bash
uv pip install torch torchvision torchaudio \
  --extra-index-url https://download.pytorch.org/whl/cpu
uv pip install huggingface-hub httpx numpy pillow pyarrow tenacity \
  pyright pytest ruff
uv pip install -e . --no-deps
```

GPU Linux。PyPI 默认 torch wheel 已自带 CUDA runtime，不需要额外索引。unsloth 对
torch 与 transformers 的约束最紧，放在最前面让解析器先满足它：

```bash
uv pip install unsloth torch torchvision torchaudio \
  transformers trl peft accelerate tensorboard \
  huggingface-hub httpx numpy pillow pyarrow tenacity \
  pyright pytest ruff
uv pip install -e . --no-deps
```

只有需要非默认 CUDA 版本时才指定索引，`cuXXX` 替换为匹配本机驱动的版本：

```bash
uv pip install torch --extra-index-url https://download.pytorch.org/whl/cuXXX
```

### 3. CraftGround

只有 CraftGround 使用项目自维护版本。核心包与 mc121 runtime 必须锁定到
`OopsYouDiedE/CraftGround` `tao-maintained` 分支的同一精确提交，不得改为 PyPI 范围依赖
或只锁分支名。提交号取自 `pyproject.toml` 的 `craftground` extra：

```bash
uv pip install -e '.[craftground]' --no-deps
uv pip install gymnasium protobuf psutil typing_extensions
```

首次创建环境会把维护版 runtime 复制到 `~/.cache/tao/` 并执行一次 Gradle 构建，因此这一步
要求 JDK 与 native 依赖已就绪。

### 4. 可视化控制台

在 Linux 或 WSL 的项目根目录启动：

```bash
source .venv/bin/activate
python -m trajectory_visualization --slots 1 --host 0.0.0.0 --port 8000 --socket-ipc
```

Windows PowerShell 可从项目根目录直接进入 WSL 2 并启动：

```powershell
wsl.exe -d Ubuntu-24.04 --cd "$PWD" bash -lc `
  'source .venv/bin/activate && exec python -m trajectory_visualization --slots 1 --host 0.0.0.0 --port 8000 --socket-ipc'
```

浏览器访问 `http://localhost:8000`。首次启动 CraftGround 可能需要完成 Gradle 构建。

### 5. Godot（可选）

Godot 环境依赖未纳入 `pyproject.toml`，单独安装：

```bash
uv pip install gymnasium stable-baselines3
```

### 6. 一次性完整安装

GPU Linux 上装齐系统依赖与全部 Python 路径：

```bash
sudo apt-get update
sudo apt-get install -y openjdk-21-jdk cmake ninja-build \
  libglew-dev libgl1-mesa-dev libglu1-mesa-dev libglfw3-dev xorg-dev \
  mesa-utils xserver-xorg-core x11-xserver-utils pciutils
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64

uv venv --python 3.13 && source .venv/bin/activate
uv pip install unsloth torch torchvision torchaudio \
  transformers trl peft accelerate tensorboard \
  huggingface-hub httpx numpy pillow pyarrow tenacity \
  gymnasium stable-baselines3 protobuf psutil typing_extensions \
  pyright pytest ruff
uv pip install -e '.[craftground]' --no-deps
```

### 7. Node 与编码 CLI 工具（可选）

项目源码不依赖 Node；以下工具用于开发期的 CLI 辅助。教师模型 CLI 后端
（`TEACHER_BACKEND=*-cli`）需要其中之一，可执行文件路径通过 `TEACHER_CLI_EXECUTABLE` 指定。

用 nvm 装当前 LTS Node，避免污染系统 apt 包：

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/master/install.sh | bash
source ~/.bashrc
nvm install --lts
nvm use --lts
node --version && npm --version
```

安装常用编码 CLI：

```bash
npm install -g @anthropic-ai/claude-code
npm install -g @openai/codex
npm install -g opencode-ai
```

各工具首次运行时自行完成登录，凭证不写入本仓库：

```bash
claude
codex login
gemini
```

### 8. 凭证与配置

教师模型通过项目根目录 `.env` 或进程环境变量配置，参照 `.env.example`。
GitHub 与 Hugging Face 按需自行登录，项目源码不包含鉴权检查模块：

```bash
cp .env.example .env
gh auth login
hf auth login
```

### 9. 安装校验

```bash
python -m pytest -q
python -m ruff check .
```

真实 CraftGround、云端模型、BC 和相对优势训练必须在各自权威环境用真实依赖验证，
上述命令只覆盖纯逻辑部分。
