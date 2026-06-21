"""握手吞吐基准：消费者每回合开销如何决定回合吞吐，且【任何开销下都不丢帧】。

在同一次 Godot 运行里，对不同"推理耗时"各跑一段，测量：回合吞吐 cycles/s、每回合平均耗时 ms、
是否无丢帧（帧号严格 +1）。结论：吞吐 ≈ 1 / (Godot步进+渲染 + 消费者开销)，随消费者变慢而下降，但始终零丢帧。
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from utils.godot_rl import shared_mem_env as E
from utils.godot_rl.launch import launch_godot, kill_godot

GODOT_LOG = os.path.join(E.PROJECT_DIR, "_godot_compare_run.log")

WARMUP_S = 0.8
MEASURE_S = 4.0


def busy_infer(ms):
    """模拟推理耗时（sleep 近似；真实场景是 GPU 前向）。"""
    if ms > 0:
        time.sleep(ms / 1000.0)


def run_load(envh, duration, infer_ms):
    """跑一段同步回合：读图+元数据 -> 推理(infer_ms) -> 发 action。"""
    zeros_cont = np.zeros((E.NUM_ENVS, E.CONT_DIM), np.float32)
    zeros_disc = np.zeros((E.NUM_ENVS, E.DISC_DIM), np.int32)
    ids = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration:
        if not envh.wait_obs(2000):
            continue
        meta = envh.read_meta()
        _ = envh.read_images()
        busy_infer(infer_ms)
        envh.send_action(zeros_cont, zeros_disc)
        ids.append(int(round(float(meta[0, E.M_FRAME]))))
    dt = time.perf_counter() - t0

    arr = np.array(ids)
    gaps = np.diff(arr) if len(arr) >= 2 else np.array([1])
    cycles = len(arr)
    lossless = bool(np.all(gaps == 1))
    return {
        "cycles_per_s": cycles / dt if dt else 0.0,
        "ms_per_cycle": 1000.0 * dt / cycles if cycles else 0.0,
        "cycles": cycles,
        "lossless": lossless,
        "max_gap": int(gaps.max()),
    }


def fmt_row(name, s):
    return (f"  {name:<14}| 吞吐 {s['cycles_per_s']:6.1f} 回合/s "
            f"| 每回合 {s['ms_per_cycle']:6.1f} ms "
            f"| 回合数 {s['cycles']:5d} "
            f"| 无丢帧 {'YES' if s['lossless'] else 'NO!'} (最大跳 {s['max_gap']})")


def main():
    E.set_timer_resolution(1)   # 仅 Windows，其它平台空操作
    print("=" * 76)
    print(" 握手吞吐基准：消费者开销 vs 回合吞吐（始终零丢帧）")
    print("=" * 76)

    log = open(GODOT_LOG, "w", encoding="utf-8", errors="replace")
    proc = launch_godot(log=log)
    zeros_cont = np.zeros((E.NUM_ENVS, E.CONT_DIM), np.float32)
    zeros_disc = np.zeros((E.NUM_ENVS, E.DISC_DIM), np.int32)
    try:
        try:
            envh = E.GodotTrainEnv(connect_timeout_s=40)
        except RuntimeError as e:
            print(f"[FAIL] {e}")
            return 1

        # 软件渲染/Linux 首帧含一次性着色器编译；预热吃掉它。
        if not envh.warmup(timeout_ms=120000, frames=2):
            print("[FAIL] 预热未收到渲染帧。")
            return 1

        scenarios = [
            ("推理 0ms", 0),
            ("推理 5ms", 5),
            ("推理 20ms", 20),
        ]

        print()
        all_lossless = True
        for title, infer_ms in scenarios:
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < WARMUP_S:
                if envh.wait_obs(2000):
                    busy_infer(infer_ms)
                    envh.send_action(zeros_cont, zeros_disc)
            s = run_load(envh, MEASURE_S, infer_ms)
            all_lossless = all_lossless and s["lossless"]
            print(fmt_row(title, s))

        envh.close()
        print()
        print("-" * 76)
        print("说明：握手下吞吐 ≈ 1 / (Godot步进+渲染 + 消费者推理)，随推理变慢线性下降；")
        print("      但每个 ObsReady 都是一帧新观测，无重复（不需去重）、无丢帧（帧号严格 +1）。")
        print(f"总判定: {'[ ALL PASS ] 各负载均零丢帧' if all_lossless else '[ FAIL ] 出现丢帧'}")
        return 0 if all_lossless else 1
    finally:
        kill_godot(proc)
        log.close()
        E.reset_timer_resolution(1)
        print("[*] 已关闭 Godot。")


if __name__ == "__main__":
    sys.exit(main())
