#!/bin/bash

# ========================================
# 系统与网络综合测试脚本 (兼容 WSL 与 sh)
# ========================================

WARN_COLOR='\033[0;31m'
NC='\033[0m'

echo "================ 1. GPU 与显存检测 ================"
# 修复 sh 下的重定向兼容性
if command -v nvidia-smi > /dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
    VRAM_GB=$(awk "BEGIN {printf \"%.2f\", $VRAM_MB/1024}")
    
    echo "GPU 型号: $GPU_NAME"
    echo "显存容量: ${VRAM_GB} GB"
    
    if awk -v vram="$VRAM_GB" 'BEGIN { if (vram < 80) exit 0; else exit 1 }'; then
        printf "${WARN_COLOR}[警告] GPU 显存容量小于 80 GB！当前为 ${VRAM_GB} GB${NC}\n"
    fi
else
    echo "未检测到 nvidia-smi，设备无 NVIDIA GPU 或未安装相关驱动。"
fi

printf "\n================ 2. CPU 配置检测 ================\n"
CPU_NAME=$(lscpu | grep "Model name" | awk -F: '{print $2}' | sed 's/^[ \t]*//')
CPU_CORES=$(nproc)
echo "CPU 型号: $CPU_NAME"
echo "逻辑核心: $CPU_CORES 核"

printf "\n================ 3. 内存 (RAM) 检测 ================\n"
TOTAL_RAM=$(free -h | awk '/^Mem:/ {print $2}')
FREE_RAM=$(free -h | awk '/^Mem:/ {print $7}')
echo "总内存容量: $TOTAL_RAM"
echo "可用内存  : $FREE_RAM"

printf "\n================ 4. 存储配置检测 ================\n"
DISK_AVAIL_GB=$(df -BG / | awk 'NR==2 {print $4}' | tr -d 'G')
DISK_TOTAL_GB=$(df -BG / | awk 'NR==2 {print $2}' | tr -d 'G')

echo "系统盘总容量: ${DISK_TOTAL_GB} GB"
echo "系统盘可用  : ${DISK_AVAIL_GB} GB"

if [ "$DISK_AVAIL_GB" -lt 100 ]; then
    printf "${WARN_COLOR}[警告] 系统盘可用存储空间小于 100 GB！当前为 ${DISK_AVAIL_GB} GB${NC}\n"
fi

printf "\n================ 5. 网速测试 (100MB) ===============\n"
TEST_URL="https://speed.hetzner.de/100MB.bin"
echo "正在测试下载速度 (最多允许 10 秒)..."

# 加入 -k 参数忽略 WSL 中可能出现的 SSL 证书问题
CURL_OUTPUT=$(curl -k -m 10 -o /dev/null -s -w "%{time_total},%{size_download},%{speed_download}" "$TEST_URL")
CURL_STATUS=$?

TIME_TOTAL=$(echo "$CURL_OUTPUT" | cut -d',' -f1)
SIZE_DL_BYTES=$(echo "$CURL_OUTPUT" | cut -d',' -f2)
SPEED_BPS=$(echo "$CURL_OUTPUT" | cut -d',' -f3)

if [ $CURL_STATUS -eq 28 ]; then
    DL_MB=$(awk "BEGIN {printf \"%.2f\", $SIZE_DL_BYTES/1024/1024}")
    SPEED_MBPS=$(awk "BEGIN {printf \"%.2f\", $SPEED_BPS/1024/1024}")
    printf "${WARN_COLOR}[警告] 下载超时！10秒内未完成 100MB 下载。${NC}\n"
    echo "已下载: $DL_MB MB"
    echo "平均网速: $SPEED_MBPS MB/s"
elif [ $CURL_STATUS -eq 0 ]; then
    SPEED_MBPS=$(awk "BEGIN {printf \"%.2f\", $SPEED_BPS/1024/1024}")
    echo "下载完成！"
    echo "耗时    : ${TIME_TOTAL} 秒"
    echo "平均网速: ${SPEED_MBPS} MB/s"
else
    printf "${WARN_COLOR}[错误] 测试失败 (curl 退出码: $CURL_STATUS)。如果依然失败，请检查是否需要配置代理。${NC}\n"
fi

printf "\n=================================================\n"