#!/bin/bash

# ========================================
# 系统与网络综合测试脚本 (兼容 WSL 与 sh)
# ========================================

WARN_COLOR='\033[0;31m'
NC='\033[0m'

echo "================ 1. GPU 与显存检测 ================"
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
    VRAM_GB=$(awk "BEGIN {printf \"%.2f\", $VRAM_MB/1024}")
    
    echo "GPU 型号: $GPU_NAME"
    echo "显存容量: ${VRAM_GB} GB"
    
    # 直接使用 MB (80GB = 81920MB) 进行原生整数对比，省去 awk 浮点比较
    if [ "$VRAM_MB" -lt 81920 ]; then
        printf "${WARN_COLOR}[警告] GPU 显存容量小于 80 GB！当前为 ${VRAM_GB} GB${NC}\n"
    fi
else
    echo "未检测到 nvidia-smi，设备无 NVIDIA GPU 或未安装相关驱动。"
fi

printf "\n================ 2. CPU 配置检测 ================\n"
# 合并 grep, awk, sed 为一句 awk 即可完成过滤和去空格
echo "CPU 型号: $(lscpu | awk -F ': +' '/Model name/ {print $2; exit}')"
echo "逻辑核心: $(nproc) 核"

printf "\n================ 3. 内存 (RAM) 检测 ================\n"
# 同时提取总内存和可用内存，减少一次 free 命令的调用
read -r TOTAL_RAM FREE_RAM <<< $(free -h | awk '/^Mem:/ {print $2, $7}')
echo "总内存容量: $TOTAL_RAM"
echo "可用内存  : $FREE_RAM"

printf "\n================ 4. 存储配置检测 ================\n"
# 利用 awk 的 int() 自动剔除 'G' 后缀，并同时提取两个值
read -r DISK_TOTAL_GB DISK_AVAIL_GB <<< $(df -BG / | awk 'NR==2 {print int($2), int($4)}')
echo "系统盘总容量: ${DISK_TOTAL_GB} GB"
echo "系统盘可用  : ${DISK_AVAIL_GB} GB"

if [ "$DISK_AVAIL_GB" -lt 100 ]; then
    printf "${WARN_COLOR}[警告] 系统盘可用存储空间小于 100 GB！当前为 ${DISK_AVAIL_GB} GB${NC}\n"
fi

printf "\n================ 5. HuggingFace 网速测试 ===============\n"
TEST_URL="https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/model.safetensors"
echo "正在从 Hugging Face 下载 133MB 测试文件 (最多允许 10 秒)..."

CURL_OUTPUT=$(curl -L -k -m 10 -o /dev/null -s -w "%{time_total},%{size_download},%{speed_download}" "$TEST_URL")
CURL_STATUS=$?

# 一次性解析三个用逗号分隔的值
IFS=, read -r TIME_TOTAL SIZE_DL_BYTES SPEED_BPS <<< "$CURL_OUTPUT"

# 提取公共计算部分 (1024*1024 = 1048576)，消除代码冗余
DL_MB=$(awk "BEGIN {printf \"%.2f\", $SIZE_DL_BYTES/1048576}")
SPEED_MBPS=$(awk "BEGIN {printf \"%.2f\", $SPEED_BPS/1048576}")

if [ $CURL_STATUS -eq 28 ]; then
    printf "${WARN_COLOR}[警告] 下载超时！10秒内未能下载完 133MB。${NC}\n"
    echo "已下载  : $DL_MB MB"
    echo "平均网速: $SPEED_MBPS MB/s"
elif [ $CURL_STATUS -eq 0 ]; then
    echo "下载完成！"
    echo "文件大小: $DL_MB MB"
    echo "耗时    : ${TIME_TOTAL} 秒"
    echo "平均网速: ${SPEED_MBPS} MB/s"
else
    printf "${WARN_COLOR}[错误] 测试失败 (curl 退出码: $CURL_STATUS)。请检查网络或是否需要代理。${NC}\n"
fi

printf "\n=================================================\n"