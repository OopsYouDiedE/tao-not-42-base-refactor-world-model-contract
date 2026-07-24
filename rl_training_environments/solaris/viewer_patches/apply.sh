#!/usr/bin/env bash
# 把 solaris 地形渲染修复补丁应用到 vendored prismarine-viewer-colalab。
#
# 用法:
#   bash apply.sh <viewer 包根目录>
# 例(engine/ 下 npm install 后):
#   bash apply.sh ../engine/node_modules/prismarine-viewer-colalab
#
# 补丁修 3 个文件(viewer/lib/{worker,worldrenderer,models}.js)让引擎正确渲染
# Minecraft 1.18+ 的负 y 世界地形。幂等:已打过的补丁会被 patch 检测并跳过。
# 详见同目录 README.md。
set -e
VIEWER="${1:?用法: bash apply.sh <prismarine-viewer-colalab 包根目录>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$VIEWER/viewer/lib" ]; then
  echo "错误: $VIEWER/viewer/lib 不存在，确认这是 prismarine-viewer-colalab 包根目录" >&2
  exit 2
fi

for f in worker worldrenderer models; do
  patch_file="$HERE/$f.js.patch"
  target="$VIEWER/viewer/lib/$f.js"
  echo "== $f.js =="
  if patch -p1 --dry-run -d "$VIEWER" < "$patch_file" >/dev/null 2>&1; then
    patch -p1 -d "$VIEWER" < "$patch_file"
    echo "  已应用"
  elif patch -p1 -R --dry-run -d "$VIEWER" < "$patch_file" >/dev/null 2>&1; then
    echo "  已是打过补丁的状态，跳过"
  else
    echo "  无法干净应用(源文件版本可能不同)，请手动核对 $patch_file" >&2
  fi
done
echo "完成。重启 viewer 进程后生效。"
