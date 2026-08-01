#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${TAO_PYTHON:-$ROOT/.venv/bin/python}"
ENTRYPOINT="${TAO_TRAIN_ENTRYPOINT:-train.bc.gemma_vision_sft}"

test -x "$PYTHON"
command -v nvidia-smi >/dev/null
nvidia-smi >/dev/null
hf auth whoami >/dev/null

"$PYTHON" -m "$ENTRYPOINT" --skip-backward "$@"
