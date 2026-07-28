#!/bin/bash
# 临时探测脚本，用完删除
echo "--- 1. PyPI wheel 实测下载 ---"
W=$(curl -m 25 -sS https://pypi.org/pypi/numpy/json \
  | tr ',' '\n' | grep -o 'https://files.pythonhosted.org[^"]*\.whl' | head -1)
echo "wheel url: $W"
curl -m 20 -L -o /dev/null -sS \
  -w "code=%{http_code} size=%{size_download} time=%{time_total} speed=%{speed_download}\n" \
  "$W" 2>&1

echo
echo "--- 2. HF LFS 重定向落到哪个主机 ---"
curl -m 20 -sSI -w "final=%{url_effective}\n" \
  https://huggingface.co/gpt2/resolve/main/pytorch_model.bin 2>&1 \
  | grep -iE '^(location|final|HTTP)' | head -5
