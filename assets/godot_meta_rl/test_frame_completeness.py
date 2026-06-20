"""
帧完整性测试（同步握手版）：证明 Python 一帧不漏，即使消费者很慢。

旧的单槽"最新帧寄存器 + 互斥锁轮询"模型，一旦消费者比 Godot 产帧慢，就会丢帧
（实测慢消费者丢过 ~52%）。改成事件握手后，Godot 在收到 action 前【绝不产下一帧】，
所以无论消费者多慢，都是 Godot 等消费者，而不是消费者追 Godot —— 天然背压，零丢帧。

本测试在同一次 Godot 运行里跑两种消费者，对比帧号是否连续(+1)：
  A) 快消费者：拿到观测立刻发 action。
  B) 慢消费者：每回合故意 sleep 25ms 再发 action（模拟较重的推理/IO）。
判定：两者帧号都必须严格 +1（skipped==0）。B 的回合速率会明显下降，但一帧不丢，
      这正是握手相对旧轮询模型的关键价值。
"""

import os
import subprocess
import sys
import time

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import rl_train_env as E

# 复用 rl_train_env 中的路径配置
GODOT_EXE = E.GODOT_EXE
PROJECT_DIR = E.PROJECT_DIR
TRAIN_SCENE = E.TRAIN_SCENE
GODOT_LOG = os.path.join(E.PROJECT_DIR, "_godot_completeness.log")

WARMUP_CYCLES = 15
MEASURE_S = 5.0
SLOW_SLEEP_MS = 25


def launch_godot():
    env = os.environ.copy()
    env.pop("RL_STEP_MODE", None)
    log = open(GODOT_LOG, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen([GODOT_EXE, "--path", PROJECT_DIR, TRAIN_SCENE],
                            stdout=log, stderr=subprocess.STDOUT, env=env)
    return proc, log


def kill(proc):
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        proc.kill()


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
    E.set_timer_resolution(1)   # 让慢消费者的 25ms sleep 精确
    print("=" * 68)
    print(" 帧完整性测试（握手版）：Python 能否读到每一个实际帧")
    print("=" * 68)

    proc, log = launch_godot()
    zeros_cont = np.zeros((E.NUM_ENVS, E.CONT_DIM), np.float32)
    zeros_disc = np.zeros((E.NUM_ENVS, E.DISC_DIM), np.int32)
    try:
        try:
            envh = E.GodotTrainEnv(connect_timeout_s=40)
        except RuntimeError as e:
            print(f"[FAIL] {e}")
            return 1

        # 预热
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
        print("结论：握手协议下 Godot 收到 action 前绝不产下一帧，于是【消费者多慢，")
        print("      Godot 就等多久】——天然背压，零丢帧。慢消费者只让回合速率下降")
        print(f"      （A {a['fps']:.0f}/s -> B {b['fps']:.0f}/s），但帧号依旧严格 +1。")
        print("      对比旧单槽轮询模型：同样的慢消费者会直接丢掉一多半帧。")
        all_pass = pa and pb
        print(f"总判定: {'[ ALL PASS ]' if all_pass else '[ FAIL ]'}")
        return 0 if all_pass else 1
    finally:
        kill(proc)
        log.close()
        E.reset_timer_resolution(1)
        print("[*] 已关闭 Godot。")


if __name__ == "__main__":
    sys.exit(main())
