#!/bin/bash
# scripts/gpu_run.sh
# 智能硬件加速包装器 - 状态记忆与免初始化复用版

set -e

# 如果没有提供命令，则退出
if [ $# -eq 0 ]; then
    echo "Usage: $0 <command> [args...]"
    exit 1
fi

# 1. 检查基础依赖
if ! command -v lspci &> /dev/null || ! command -v xdpyinfo &> /dev/null; then
    echo "[gpu_run] 正在安装基础检测工具 (pciutils, x11-utils)..."
    sudo apt-get update && sudo apt-get install -y pciutils x11-utils
fi

# 2. 尝试探测 GPU
NVIDIA_BUS_ID=""
if lspci | grep -i nvidia &> /dev/null; then
    # 提取第一个 NVIDIA 显卡的 BusID (格式为 01:00.0)
    RAW_BUS_ID=$(lspci | grep -i nvidia | grep -E "VGA|3D" | head -n 1 | awk '{print $1}')
    if [ -n "$RAW_BUS_ID" ]; then
        # 转换为 Xorg 格式 (十六进制转十进制) PCI:bus:dev:func
        BUS=$(echo $RAW_BUS_ID | cut -d: -f1)
        DEV_FUNC=$(echo $RAW_BUS_ID | cut -d: -f2)
        DEV=$(echo $DEV_FUNC | cut -d. -f1)
        FUNC=$(echo $DEV_FUNC | cut -d. -f2)
        
        # 将16进制转10进制
        BUS_DEC=$((16#$BUS))
        DEV_DEC=$((16#$DEV))
        FUNC_DEC=$((16#$FUNC))
        NVIDIA_BUS_ID="PCI:${BUS_DEC}:${DEV_DEC}:${FUNC_DEC}"
        echo "[gpu_run] 检测到 NVIDIA GPU: $NVIDIA_BUS_ID"
    fi
fi

# 3. 配置文件复用与生成
XORG_CONF="/etc/X11/xorg.conf.headless"
if [ -n "$NVIDIA_BUS_ID" ]; then
    if [ ! -f "$XORG_CONF" ] || ! grep -q "$NVIDIA_BUS_ID" "$XORG_CONF"; then
        echo "[gpu_run] 正在生成/更新 Xorg 配置文件 $XORG_CONF ..."
        sudo bash -c "cat > $XORG_CONF <<EOF
Section \"ServerLayout\"
    Identifier     \"Layout0\"
    Screen      0  \"Screen0\" 0 0
EndSection

Section \"ServerFlags\"
    Option \"AllowMouseOpenFail\" \"true\"
    Option \"AutoAddDevices\" \"false\"
    Option \"AutoEnableDevices\" \"false\"
EndSection

Section \"Device\"
    Identifier     \"Device0\"
    Driver         \"nvidia\"
    VendorName     \"NVIDIA Corporation\"
    BusID          \"$NVIDIA_BUS_ID\"
EndSection

Section \"Screen\"
    Identifier     \"Screen0\"
    Device         \"Device0\"
    Monitor        \"Monitor0\"
    DefaultDepth    24
    Option         \"UseDisplayDevice\" \"None\"
    Option         \"Virtual\" \"1280 1024\"
    SubSection     \"Display\"
        Depth       24
    EndSubSection
EndSection

Section \"Monitor\"
    Identifier     \"Monitor0\"
    VendorName     \"Unknown\"
    ModelName      \"Unknown\"
    HorizSync       28.0 - 33.0
    VertRefresh     43.0 - 72.0
    Option         \"DPMS\"
EndSection
EOF"
    else
        echo "[gpu_run] 发现有效的 $XORG_CONF 配置，跳过生成。"
    fi
fi

# 4. 显示服务复用拉起
export DISPLAY=":1"
USING_GPU=false

if [ -n "$NVIDIA_BUS_ID" ]; then
    # 检查 :1 是否已经在运行并且可以正常接受连接
    if pgrep -f "Xorg :1" > /dev/null && xdpyinfo -display :1 >/dev/null 2>&1; then
        echo "[gpu_run] 发现常驻 GPU 渲染服务 Xorg :1 并且运行正常，直接复用！"
        USING_GPU=true
    else
        echo "[gpu_run] 正在启动常驻 GPU 渲染服务 Xorg :1 ..."
        # 清理可能残留的死锁和无响应进程
        sudo pkill -f "Xorg :1" || true
        sudo rm -f /tmp/.X1-lock /tmp/.X11-unix/X1
        sudo Xorg :1 -config $XORG_CONF -nolisten tcp > /dev/null 2>&1 &
        
        # 等待启动并强验证是否能建立显示连接
        for i in {1..30}; do
            if [ -S "/tmp/.X11-unix/X1" ] && xdpyinfo -display :1 >/dev/null 2>&1; then
                USING_GPU=true
                echo "[gpu_run] GPU 渲染服务启动成功并已通过可用性验证！"
                break
            fi
            sleep 0.2
        done
        
        if [ "$USING_GPU" = false ]; then
            echo "[gpu_run] 严重警告: Xorg :1 启动失败或未能响应渲染请求。"
            echo "[gpu_run] 请检查 /var/log/Xorg.1.log 了解显卡报错详情。"
        fi
    fi
fi

# 5. 降级机制
if [ "$USING_GPU" = false ]; then
    echo "[gpu_run] 警告: 未使用 GPU 渲染。降级使用 Xvfb CPU 软渲染。"
    export DISPLAY=":99"
    if pgrep -f "Xvfb :99" > /dev/null && xdpyinfo -display :99 >/dev/null 2>&1; then
        echo "[gpu_run] 发现常驻 Xvfb :99，直接复用！"
    else
        if ! command -v Xvfb &> /dev/null; then
            echo "[gpu_run] 正在安装 xvfb..."
            sudo apt-get update && sudo apt-get install -y xvfb
        fi
        echo "[gpu_run] 正在启动常驻 Xvfb :99 ..."
        sudo pkill -f "Xvfb :99" || true
        sudo rm -f /tmp/.X99-lock /tmp/.X11-unix/X99
        # 注意: 出于安全考虑，Xvfb 不需要 sudo 启动，但如果在 root 下运行则无所谓
        Xvfb :99 -screen 0 640x360x24 -nolisten tcp > /dev/null 2>&1 &
        
        # 验证 Xvfb
        for i in {1..30}; do
            if [ -S "/tmp/.X11-unix/X99" ] && xdpyinfo -display :99 >/dev/null 2>&1; then
                echo "[gpu_run] Xvfb 启动成功并验证通过！"
                break
            fi
            sleep 0.2
        done
    fi
fi

echo "[gpu_run] ======================================"
echo "[gpu_run] 环境就绪，DISPLAY=$DISPLAY，开始执行目标程序"
echo "[gpu_run] 命令: $@"
echo "[gpu_run] ======================================"

# 为 Python 模块解析自动增加当前目录，避免 ModuleNotFoundError
export PYTHONPATH=".:$PYTHONPATH"

# 执行真正的命令
exec "$@"
