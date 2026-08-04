#!/usr/bin/env bash
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
python scripts/check_environment.py --accelerator cuda

java_version="$(java -version 2>&1 | head -n 1)"
if [[ "$java_version" != *'version "21.'* ]]; then
  echo "OpenJDK 21 is required, found: $java_version" >&2
  exit 1
fi
python -c 'import craftground; import torch; assert torch.cuda.is_available()'
