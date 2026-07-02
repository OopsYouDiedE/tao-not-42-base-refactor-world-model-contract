#!/usr/bin/env python3
"""系统运行情况监控小工具(GPU/CPU/内存,低利用率告警)。

每 interval 秒采样一次:GPU 利用率/显存(nvidia-smi)、CPU 利用率(/proc/stat)、
内存(/proc/meminfo),追加写入 CSV;GPU 利用率的滑窗均值低于阈值时向 stdout
打 [LOW-UTIL] 告警行(供上层 tail/Monitor 抓取)。GPU 利用率是突发性的
(见 knowledge/dreamer.md §2.5),故告警看滑窗均值而非单点。

使用方法:
    python scripts/sys_monitor.py --interval 5 --low-util 30 --window 12 \
        --csv runs/logs/sys_monitor.csv
"""
import argparse
import csv
import os
import subprocess
import time
from collections import deque


def gpu_stats():
    """nvidia-smi → (util %, mem_used MB, mem_total MB);无 GPU 返回 None。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        util, used, total = [float(x) for x in out.split(",")]
        return util, used, total
    except Exception:
        return None


def cpu_times():
    with open("/proc/stat") as f:
        parts = f.readline().split()[1:]
    vals = [int(x) for x in parts]
    idle = vals[3] + vals[4]                     # idle + iowait
    return idle, sum(vals)


def mem_stats():
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":")
            info[k] = int(v.split()[0])
    total = info["MemTotal"] / 1e6
    avail = info["MemAvailable"] / 1e6
    return total - avail, total                  # used GB, total GB


def main():
    p = argparse.ArgumentParser(description="GPU/CPU/内存监控与低利用率告警")
    p.add_argument("--interval", type=float, default=5.0, help="采样间隔(秒)")
    p.add_argument("--low-util", type=float, default=30.0,
                   help="GPU 利用率滑窗均值低于该值(%%)时告警")
    p.add_argument("--window", type=int, default=12, help="滑窗采样点数")
    p.add_argument("--csv", default="runs/logs/sys_monitor.csv")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    new = not os.path.exists(args.csv)
    fout = open(args.csv, "a", newline="")
    writer = csv.writer(fout)
    if new:
        writer.writerow(["time", "gpu_util", "gpu_mem_used_mb", "gpu_mem_total_mb",
                         "cpu_util", "ram_used_gb", "ram_total_gb"])

    win = deque(maxlen=args.window)
    prev_idle, prev_total = cpu_times()
    warned = False
    print(f"[sys_monitor] 每 {args.interval}s 采样 → {args.csv};"
          f"GPU 滑窗均值 < {args.low_util}% 告警", flush=True)
    while True:
        time.sleep(args.interval)
        idle, total = cpu_times()
        cpu = 100.0 * (1.0 - (idle - prev_idle) / max(total - prev_total, 1))
        prev_idle, prev_total = idle, total
        ram_used, ram_total = mem_stats()
        g = gpu_stats()
        gu, gm, gt = g if g else (float("nan"),) * 3
        writer.writerow([time.strftime("%H:%M:%S"), gu, gm, gt,
                         round(cpu, 1), round(ram_used, 1), round(ram_total, 1)])
        fout.flush()
        if g:
            win.append(gu)
            avg = sum(win) / len(win)
            if len(win) == win.maxlen and avg < args.low_util and not warned:
                print(f"[LOW-UTIL] GPU 利用率滑窗均值 {avg:.0f}% < {args.low_util}% "
                      f"(CPU {cpu:.0f}%, 显存 {gm:.0f}/{gt:.0f}MB)——"
                      f"检查是否数据饥饿/batch 偏小/在等环境", flush=True)
                warned = True
            elif avg >= args.low_util and warned:
                print(f"[UTIL-OK] GPU 利用率恢复到 {avg:.0f}%", flush=True)
                warned = False


if __name__ == "__main__":
    main()
