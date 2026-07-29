#!/bin/bash

# ========================================
# 系统与网络综合测试脚本 (兼容 WSL 与 sh)
# ========================================

WARN_COLOR='\033[0;31m'
NC='\033[0m'

echo "================ 1. GPU 与显存检测 ================"
# 恢复标准 sh 兼容的重定向用法，避免使用 &>
if command -v nvidia-smi > /dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
    VRAM_GB=$(awk "BEGIN {printf \"%.2f\", $VRAM_MB/1024}")
    
    echo "GPU 型号: $GPU_NAME"
    echo "显存容量: ${VRAM_GB} GB"
    
    # 整数对比
    if [ "$VRAM_MB" -lt 81920 ] 2>/dev/null; then
        printf "${WARN_COLOR}[警告] GPU 显存容量小于 80 GB！当前为 ${VRAM_GB} GB${NC}\n"
    fi
else
    echo "未检测到 nvidia-smi，设备无 NVIDIA GPU 或未安装相关驱动。"
fi

printf "\n================ 2. CPU 配置检测 ================\n"
# 优化 awk，兼容性更好，去除了之前可能导致旧版 awk 报错的正则分隔符
CPU_NAME=$(lscpu | awk -F':' '/Model name/ {sub(/^[ \t]+/, "", $2); print $2; exit}')
CPU_CORES=$(nproc)
echo "CPU 型号: ${CPU_NAME:-"未知"}"
echo "逻辑核心: ${CPU_CORES} 核"

printf "\n================ 3. 内存 (RAM) 检测 ================\n"
# 移除 <<< 语法，改回两次赋值以保证 sh 绝对兼容
TOTAL_RAM=$(free -h | awk '/^Mem:/ {print $2}')
FREE_RAM=$(free -h | awk '/^Mem:/ {print $7}')
echo "总内存容量: $TOTAL_RAM"
echo "可用内存  : $FREE_RAM"

printf "\n================ 4. 存储配置检测 ================\n"
# 同样为了 sh 兼容，拆分赋值但依然用 awk 的 int() 简化处理
DISK_TOTAL_GB=$(df -BG / | awk 'NR==2 {print int($2)}')
DISK_AVAIL_GB=$(df -BG / | awk 'NR==2 {print int($4)}')

echo "系统盘总容量: ${DISK_TOTAL_GB} GB"
echo "系统盘可用  : ${DISK_AVAIL_GB} GB"

if [ "$DISK_AVAIL_GB" -lt 100 ] 2>/dev/null; then
    printf "${WARN_COLOR}[警告] 系统盘可用存储空间小于 100 GB！当前为 ${DISK_AVAIL_GB} GB${NC}\n"
fi

printf "\n================ 5. HuggingFace 网速测试 ===============\n"
TEST_URL="https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/model.safetensors"
echo "正在从 Hugging Face 下载 133MB 测试文件 (最多允许 10 秒)..."

CURL_OUTPUT=$(curl -L -k -m 10 -o /dev/null -s -w "%{time_total},%{size_download},%{speed_download}" "$TEST_URL")
CURL_STATUS=$?

# 恢复使用 cut，完全兼容 sh 避免 read 分片时报错
TIME_TOTAL=$(echo "$CURL_OUTPUT" | cut -d',' -f1)
SIZE_DL_BYTES=$(echo "$CURL_OUTPUT" | cut -d',' -f2)
SPEED_BPS=$(echo "$CURL_OUTPUT" | cut -d',' -f3)

# 统一提取浮点计算
DL_MB=$(awk "BEGIN {printf \"%.2f\", ${SIZE_DL_BYTES:-0}/1048576}")
SPEED_MBPS=$(awk "BEGIN {printf \"%.2f\", ${SPEED_BPS:-0}/1048576}")

if [ "$CURL_STATUS" -eq 28 ]; then
    printf "${WARN_COLOR}[警告] 下载超时！10秒内未能下载完 133MB。${NC}\n"
    echo "已下载  : $DL_MB MB"
    echo "平均网速: $SPEED_MBPS MB/s"
elif [ "$CURL_STATUS" -eq 0 ]; then
    echo "下载完成！"
    echo "文件大小: $DL_MB MB"
    echo "耗时    : ${TIME_TOTAL} 秒"
    echo "平均网速: ${SPEED_MBPS} MB/s"
else
    printf "${WARN_COLOR}[错误] 测试失败 (curl 退出码: $CURL_STATUS)。请检查网络或是否需要代理。${NC}\n"
fi

printf "\n=================================================\n"