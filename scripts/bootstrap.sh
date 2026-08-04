#!/usr/bin/env bash
# 通用一键安装：按锁定版本或兼容范围最新版本安装 Python 依赖，并做安装后验证。
#
# 用法：
#   bash scripts/bootstrap.sh                      # 锁定版本，自动选择 cpu / cuda
#   bash scripts/bootstrap.sh --latest             # 兼容范围内最新版本
#   bash scripts/bootstrap.sh --accelerator cpu    # 显式指定后端
#   bash scripts/bootstrap.sh --accelerator cuda
#
# 本项目源码不管理鉴权和系统环境检查。GitHub 与 Hugging Face 按需自行执行 `gh auth login`
# 与 `hf auth login`；教师模型 API 通过根目录 `.env` 或进程环境变量配置。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
ACCELERATOR="auto"
MODE="locked"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --latest)
      MODE="latest"
      shift
      ;;
    --locked)
      MODE="locked"
      shift
      ;;
    --accelerator)
      ACCELERATOR="${2:?--accelerator 需要 cpu 或 cuda}"
      shift 2
      ;;
    *)
      echo "未知参数：$1" >&2
      exit 2
      ;;
  esac
done

if [[ "$ACCELERATOR" == "auto" ]]; then
  # 依据已安装 PyTorch 的真实 CUDA 可用性判断；尚未安装 PyTorch 时按 cpu 处理。
  ACCELERATOR="$("$PYTHON" - <<'PYTHON'
try:
    import torch
except ImportError:
    print("cpu")
else:
    print("cuda" if torch.cuda.is_available() else "cpu")
PYTHON
)"
fi
if [[ "$ACCELERATOR" != "cpu" && "$ACCELERATOR" != "cuda" ]]; then
  echo "--accelerator 必须是 cpu 或 cuda，收到：$ACCELERATOR" >&2
  exit 2
fi

cd "$REPO_ROOT"
if [[ "$MODE" == "latest" ]]; then
  if [[ "$ACCELERATOR" == "cuda" ]]; then
    "$PYTHON" -m pip install -e '.[cuda]'
  else
    "$PYTHON" -m pip install -e .
  fi
else
  "$PYTHON" -m pip install -r "requirements/locked-$ACCELERATOR.txt"
  "$PYTHON" -m pip install --no-deps -e .
fi

if [[ "$ACCELERATOR" == "cpu" ]]; then
  # CPU 路径从 PyTorch 官方 CPU 索引安装，避免引入 CUDA wheel。
  if [[ "$MODE" == "latest" ]]; then
    "$PYTHON" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  else
    "$PYTHON" -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
  fi
fi

ACCELERATOR="$ACCELERATOR" "$PYTHON" - <<'PYTHON'
import os
import platform
import sys

assert sys.version_info >= (3, 11), f"需要 Python 3.11 或更高版本，当前为 {platform.python_version()}"

import huggingface_hub  # noqa: F401
import httpx  # noqa: F401
import numpy  # noqa: F401
import PIL  # noqa: F401
import torch

print(f"Python {platform.python_version()} at {sys.executable}")
print(f"PyTorch {torch.__version__} CPU 计算校验 {torch.ones(2).sum().item()}")
if os.environ["ACCELERATOR"] == "cuda":
    assert torch.cuda.is_available(), "PyTorch 无法使用 CUDA"
    print(f"CUDA {torch.version.cuda} on {torch.cuda.get_device_name(0)}")
    print(f"CUDA 计算校验 {torch.ones(2, device='cuda').sum().item()}")
PYTHON
