"""
握手吞吐基准（取代旧的 NAIVE vs FAST 去重对比）。

为什么不再做"去重对比"：旧版比的是"单线程轮询+内联推理" vs "读取线程+去重+队列"，
其收益全部来自【消除重复帧的重复拷贝/推理】。而事件握手下每个 ObsReady 恰好一帧新观测，
根本不存在重复帧，也没有轮询/锁竞争——去重这件事失去了对象。

握手下真正值得测的是：消费者每回合的开销如何决定回合吞吐，且【任何开销下都不丢帧】。
本基准在同一次 Godot 运行里，对不同"推理耗时"各跑一段，测量：
  - 回合吞吐 cycles/s（= 唯一帧/s，握手下二者相等）
  - 每回合平均耗时 ms
  - 是否无丢帧（帧号严格 +1）
结论：吞吐 ≈ 1 / (Godot步进+渲染 + 消费者开销)，随消费者变慢而下降，但始终零丢帧。
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
GODOT_LOG = os.path.join(E.PROJECT_DIR, "_godot_compare_run.log")

WARMUP_S = 0.8
MEASURE_S = 4.0


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
    E.set_timer_resolution(1)
    print("=" * 76)
    print(" 握手吞吐基准：消费者开销 vs 回合吞吐（始终零丢帧）")
    print("=" * 76)

    proc, log = launch_godot()
    zeros_cont = np.zeros((E.NUM_ENVS, E.CONT_DIM), np.float32)
    zeros_disc = np.zeros((E.NUM_ENVS, E.DISC_DIM), np.int32)
    try:
        try:
            envh = E.GodotTrainEnv(connect_timeout_s=40)
        except RuntimeError as e:
            print(f"[FAIL] {e}")
            return 1

        scenarios = [
            ("推理 0ms", 0),
            ("推理 5ms", 5),
            ("推理 20ms", 20),
        ]

        print()
        all_lossless = True
        for title, infer_ms in scenarios:
            # 预热
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
        kill(proc)
        log.close()
        E.reset_timer_resolution(1)
        print("[*] 已关闭 Godot。")


if __name__ == "__main__":
    sys.exit(main())
