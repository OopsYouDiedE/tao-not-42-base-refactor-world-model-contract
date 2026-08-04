#!/usr/bin/env bash
# 在 WSL 2 Ubuntu 上运行「轨迹完全由云端大模型生成」的相对优势比较流程。
#
# 全部 arm 使用同一云端模型和同一提示词，构成同策略多分支样本；每个 arm 的模型生成轮次上限由
# MAX_GENERATIONS 控制。所有 arm 结束后由 run_four_teacher_trajectories 统一执行逐轨迹审核、
# build_comparison_group 相对优势计算和 review_comparison 复核。
#
# 凭据契约：脚本不读取 ~/.codex/auth.json 或任何 CLI 私有凭据文件，TEACHER_API_KEY 必须由调用方
# 显式导出。gpt-5.6 系列只支持 responses 协议，因此 TEACHER_WIRE_API 默认为 responses。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${TAO_VENV:-$HOME/.venvs/tao-not-42-base-refactor-world-model-contract}"
PYTHON="$VENV/bin/python"

: "${TEACHER_API_KEY:?必须先导出 TEACHER_API_KEY}"
export TEACHER_BACKEND="${TEACHER_BACKEND:-openai-api}"
export TEACHER_API_URL="${TEACHER_API_URL:-https://www.packyapi.ai/v1}"
export TEACHER_MODEL="${TEACHER_MODEL:-gpt-5.6-terra}"
export TEACHER_WIRE_API="${TEACHER_WIRE_API:-responses}"
export TEACHER_TIMEOUT_SECONDS="${TEACHER_TIMEOUT_SECONDS:-300}"

# 单个 Minecraft 客户端约需 1.5 GB 堆；ENVIRONMENT_COUNT 必须按可用内存设置，
# 超出环境槽位的 arm 由 ParallelRolloutRunner 在池外排队，不会同时占用内存。
export CRAFTGROUND_JVM_MAX_MEMORY="${CRAFTGROUND_JVM_MAX_MEMORY:-1500m}"
PORT_BASE="${PORT_BASE:-18700}"
TRAJECTORY_COUNT="${TRAJECTORY_COUNT:-8}"
MAX_GENERATIONS="${MAX_GENERATIONS:-10}"
ACTION_BUDGET_TICKS="${ACTION_BUDGET_TICKS:-128}"
ENVIRONMENT_COUNT="${ENVIRONMENT_COUNT:-4}"
OUTPUT="${OUTPUT:-runs/cloud-relative-advantage-$(date -u +%Y%m%d-%H%M%S)}"

cd "$REPO_ROOT"
ARGUMENTS=(
  --output "$OUTPUT"
  --trajectory-count "$TRAJECTORY_COUNT"
  --max-generations "$MAX_GENERATIONS"
  --action-budget-ticks "$ACTION_BUDGET_TICKS"
  --port-base "$PORT_BASE"
  --environment-count "$ENVIRONMENT_COUNT"
  --socket-ipc
)
if [[ -n "${BASELINE_WORLD:-}" ]]; then
  ARGUMENTS+=(--baseline-world "$BASELINE_WORLD")
fi

PYTHONPATH="$REPO_ROOT" \
  "$PYTHON" -m environment_validation_tools.run_four_teacher_trajectories "${ARGUMENTS[@]}"
