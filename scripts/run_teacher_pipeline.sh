#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${TAO_PYTHON:-$ROOT/.venv/bin/python}"
CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
RUNTIME="${TAO_CRAFTGROUND_RUNTIME:-$HOME/.cache/tao/craftground-runtime-patched}"
RUN_ID="${TAO_RUN_ID:-codex-teacher-batch8-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT="${TAO_OUTPUT:-$ROOT/runs/$RUN_ID}"
PREFLIGHT="$OUTPUT/preflight"

test -x "$PYTHON"
command -v codex >/dev/null
command -v xvfb-run >/dev/null
command -v java >/dev/null
command -v nvidia-smi >/dev/null
gh auth status >/dev/null
hf auth whoami >/dev/null
nvidia-smi >/dev/null

mkdir -p "$PREFLIGHT"
export CODEX_HOME
eval "$("$PYTHON" -m tools.export_codex_api_env --codex-home "$CODEX_HOME")"

"$PYTHON" -m tools.prepare_craftground_runtime --target "$RUNTIME"

xvfb-run -a "$PYTHON" -m game_environment.verify_memory_snapshot \
  --runtime "$RUNTIME" \
  --output "$PREFLIGHT/memory-snapshot.json"

xvfb-run -a "$PYTHON" -m tools.audits.codex_teacher_batch8 \
  --runtime "$RUNTIME" \
  --output "$OUTPUT/teacher" \
  --codex-model "$API_MODEL" \
  --codex-executable "$(command -v codex)" \
  --codex-timeout-seconds "${TAO_CODEX_TIMEOUT_SECONDS:-240}" \
  --codex-max-attempts "${TAO_CODEX_MAX_ATTEMPTS:-3}"

printf 'Teacher pipeline completed: %s\n' "$OUTPUT/teacher"
