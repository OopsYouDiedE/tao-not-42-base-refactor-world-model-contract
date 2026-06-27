#!/usr/bin/env python3
"""
监控训练脚本：每2分钟检查一次网络和CPU活动
"""
import subprocess
import time
import psutil
import sys
from datetime import datetime

def get_network_io():
    """获取网络IO统计"""
    try:
        net_io = psutil.net_io_counters()
        return net_io.bytes_sent + net_io.bytes_recv
    except:
        return 0

def get_cpu_percent(interval=1):
    """获取CPU使用率"""
    try:
        return psutil.cpu_percent(interval=interval)
    except:
        return 0

def check_process_health():
    """检查Python进程的健康状态"""
    try:
        result = subprocess.run(
            "ps aux | grep 'test_craftground_a2c.py' | grep -v grep",
            shell=True, capture_output=True, text=True
        )
        return len(result.stdout.strip()) > 0
    except:
        return False

def main():
    print("=" * 70)
    print("开始启动训练脚本并监控...")
    print("=" * 70)

    # 启动训练脚本
    print("\n[启动] 运行 test_craftground_a2c.py...")
    proc = subprocess.Popen(
        ["python", "test_craftground_a2c.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    print(f"[PID] {proc.pid}")

    last_network_io = get_network_io()
    check_count = 0
    consecutive_idle = 0

    try:
        while proc.poll() is None:  # 进程仍在运行
            time.sleep(120)  # 等待2分钟
            check_count += 1

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[检查 #{check_count}] {current_time}")
            print("-" * 70)

            # 检查CPU使用率
            cpu_percent = get_cpu_percent(interval=2)
            print(f"  CPU使用率: {cpu_percent:.1f}%")

            # 检查网络活动
            current_network_io = get_network_io()
            network_delta = current_network_io - last_network_io
            last_network_io = current_network_io
            print(f"  网络流量: {network_delta / (1024*1024):.2f} MB")

            # 检查进程存活
            is_alive = check_process_health()
            print(f"  进程状态: {'✓ 运行中' if is_alive else '✗ 已停止'}")

            # 判断健康状态
            if cpu_percent > 5 or network_delta > 1024 * 100:  # CPU > 5% 或网络 > 100KB
                consecutive_idle = 0
                print(f"\n  ✓ 正常运行（有CPU活动或网络活动）")
            else:
                consecutive_idle += 1
                print(f"\n  ⚠️   无活动 ({consecutive_idle}次)")

                if consecutive_idle >= 2:
                    print(f"\n  ✗ 问题: 连续2次检查无网络和CPU活动，可能卡死")
                    print(f"  建议: 检查日志或重启训练")
                    break

            # 显示进程内存占用
            try:
                p = psutil.Process(proc.pid)
                mem_info = p.memory_info()
                print(f"  内存占用: {mem_info.rss / (1024*1024):.1f} MB")
            except:
                pass

        # 等待进程完成
        print(f"\n[等待] 等待进程完成...")
        proc.wait(timeout=30)
        print(f"\n✓ 训练完成！")

    except subprocess.TimeoutExpired:
        print(f"\n✗ 进程未在规定时间内完成")
        proc.kill()
    except KeyboardInterrupt:
        print(f"\n[中断] 用户中断")
        proc.kill()
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        proc.kill()

if __name__ == "__main__":
    main()
