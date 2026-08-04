#!/usr/bin/env bash
# GPU Linux 一键安装：系统依赖、Python 依赖和安装后真实环境验证。
#
# 本项目源码不管理鉴权和系统环境检查。GitHub 与 Hugging Face 按需自行执行 `gh auth login`
# 与 `hf auth login`；教师模型 API 通过根目录 `.env` 或进程环境变量配置。
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "CraftGround GPU environment requires Linux; use WSL 2 when starting from Windows." >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi
if ! command -v apt-get >/dev/null 2>&1; then
  echo "This installer currently supports apt-based Ubuntu environments." >&2
  exit 1
fi

apt-get update
apt-get install -y \
  openjdk-21-jdk \
  libgl1-mesa-dev \
  libglu1-mesa-dev \
  libglew-dev \
  libpng-dev \
  zlib1g-dev \
  xvfb \
  xauth

uv pip install --system -e '.[cuda,craftground,dev]'

java_version="$(java -version 2>&1 | head -n 1)"
if [[ "$java_version" != *'version "21.'* ]]; then
  echo "OpenJDK 21 is required, found: $java_version" >&2
  exit 1
fi

python - <<'PYTHON'
import platform
import sys

assert sys.version_info >= (3, 11), f"需要 Python 3.11 或更高版本，当前为 {platform.python_version()}"

import craftground  # noqa: F401
import torch

assert torch.cuda.is_available(), "PyTorch 无法使用 CUDA"
print(f"Python {platform.python_version()} at {sys.executable}")
print(f"PyTorch {torch.__version__} CUDA {torch.version.cuda} on {torch.cuda.get_device_name(0)}")
print(f"CUDA 计算校验 {torch.ones(2, device='cuda').sum().item()}")
PYTHON
