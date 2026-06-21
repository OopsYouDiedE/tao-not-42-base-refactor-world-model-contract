"""帧完整性测试（同步握手版）：证明 Python 一帧不漏，即使消费者很慢。

握手下 Godot 在收到 action 前【绝不产下一帧】，故无论消费者多慢都是 Godot 等消费者——天然背压、零丢帧。
本测试在同一次 Godot 运行里跑两种消费者，对比帧号是否连续(+1)：
  A) 快消费者：拿到观测立刻发 action。
  B) 慢消费者：每回合故意 sleep 25ms 再发 action（模拟较重的推理/IO）。
判定：两者帧号都必须严格 +1（skipped==0）。
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

GODOT_LOG = os.path.join(E.PROJECT_DIR, "_godot_completeness.log")

WARMUP_CYCLES = 15
MEASURE_S = 5.0
SLOW_SLEEP_MS = 25


def run_phase(envh, duration, consumer_sleep_ms):
    """同步回合循环：记录每回合帧号；可选地在发 action 前 sleep 模拟慢消费者。"""
    zeros_cont = np.zeros((E.NUM_ENVS, E.CONT_DIM), np.float32)
    zeros_disc = np.zeros((E.NUM_ENVS, E.DISC_DIM), np.int32)
    ids = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration:
        if not envh.wait_obs(2000):
            continue
        meta = envh.read_meta()
        _ = envh.read_images()                  # 真实消费者会读整图，计入耗时
        if consumer_sleep_ms:
            time.sleep(consumer_sleep_ms / 1000.0)
        envh.send_action(zeros_cont, zeros_disc)
        ids.append(int(round(float(meta[0, E.M_FRAME]))))
    dt = time.perf_counter() - t0

    arr = np.array(ids)
    gaps = np.diff(arr) if len(arr) >= 2 else np.array([1])
    breaks = gaps[gaps > 1]
    captured = len(arr)
    produced = int(arr[-1] - arr[0] + 1) if len(arr) >= 2 else captured
    skipped = int((breaks - 1).sum()) if len(breaks) else 0
    return {
        "dt": dt,
        "captured": captured,
        "produced": produced,
        "skipped": skipped,
        "n_breaks": int(len(breaks)),
        "max_gap": int(gaps.max()),
        "fps": captured / dt if dt else 0.0,
    }


def report(title, s):
    lossless = (s["skipped"] == 0)
    print(f"### {title}")
    print(f"    Godot 产出帧数(按帧号跨度): {s['produced']}")
    print(f"    Python 捕获回合数          : {s['captured']}   (~{s['fps']:.1f} 回合/s)")
    print(f"    跳号次数 / 漏掉帧数        : {s['n_breaks']} 次 / {s['skipped']} 帧   (最大一次跳 {s['max_gap']})")
    print(f"    => {'PASS  无丢帧（帧号严格 +1）' if lossless else 'FAIL  存在丢帧！'}")
    print()
    return lossless


def main():
    E.set_timer_resolution(1)   # 让慢消费者的 25ms sleep 精确（仅 Windows，其它平台空操作）
    print("=" * 68)
    print(" 帧完整性测试（握手版）：Python 能否读到每一个实际帧")
    print("=" * 68)

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
        for _ in range(WARMUP_CYCLES):
            if envh.wait_obs(2000):
                envh.send_action(zeros_cont, zeros_disc)

        a = run_phase(envh, MEASURE_S, 0)
        b = run_phase(envh, MEASURE_S, SLOW_SLEEP_MS)
        envh.close()

        print()
        pa = report("情形 A：快消费者（拿到即发 action）", a)
        pb = report(f"情形 B：慢消费者（发 action 前 sleep {SLOW_SLEEP_MS}ms）", b)
        print("-" * 68)
        print("结论：握手协议下 Godot 收到 action 前绝不产下一帧，于是【消费者多慢，Godot 就等多久】")
        print(f"      ——天然背压，零丢帧。慢消费者只让回合速率下降（A {a['fps']:.0f}/s -> B {b['fps']:.0f}/s），")
        print("      但帧号依旧严格 +1。")
        all_pass = pa and pb
        print(f"总判定: {'[ ALL PASS ]' if all_pass else '[ FAIL ]'}")
        return 0 if all_pass else 1
    finally:
        kill_godot(proc)
        log.close()
        E.reset_timer_resolution(1)
        print("[*] 已关闭 Godot。")


if __name__ == "__main__":
    sys.exit(main())
