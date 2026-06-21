"""验证两种步进模式 + action 门控 + 无丢帧（同步 lock-step 协议）。

对每种模式启动一次 Godot，跑若干同步回合，校验：
  [门控] 不发 action 时，Godot 不产出新观测（证明"action 返回前绝不步进"）。
  [无丢] renderFrameCount 连续递增(+1)，无跳号。
  [步数] Fixed: 每帧物理步数恒为 X；Decoupled: 步数在 [1,Max] 间变化且最小>=1。
  [一致] 同一帧里 40 个环境的步数一致（方差 0）。
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from utils.godot_rl import shared_mem_env as E
from utils.godot_rl.launch import launch_godot, kill_godot

CYCLES = 240


def run_mode(title, mode, fixed_steps, max_steps, log_path):
    print(f"\n### {title}  (RL_STEP_MODE={mode}, X={fixed_steps}, Max={max_steps})")
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    proc = launch_godot(log=log, extra_env={
        "RL_STEP_MODE": mode, "RL_FIXED_STEPS": fixed_steps, "RL_MAX_STEPS": max_steps})
    zeros_cont = np.zeros((E.NUM_ENVS, E.CONT_DIM), np.float32)
    zeros_disc = np.zeros((E.NUM_ENVS, E.DISC_DIM), np.int32)
    frame_ids = []
    steps_seq = []
    cross_env_var = 0.0
    gated = False
    try:
        envh = E.GodotTrainEnv(connect_timeout_s=40)
        # 软件渲染/Linux 着色器惰性编译：除基础变体外，聚光灯点亮时还会再编译光照变体（又一次秒级停顿）。
        # 多预热几帧把这些一次性编译吃掉，避免门控/首帧判定被编译停顿干扰。
        assert envh.warmup(timeout_ms=120000, frames=6), "预热未收到渲染帧"

        # 拿到一帧观测（仍给长超时，兜住可能尚未触发的光照编译）。
        assert envh.wait_obs(60000), "未收到观测"
        _ = envh.read_meta()

        # [门控] 不发 action，看 Godot 是否"卡住不产新帧"
        gated = not envh.wait_obs(500)   # 期望超时(无新帧)
        envh.send_action(zeros_cont, zeros_disc)  # 解除，进入正常回合

        for _ in range(CYCLES):
            if not envh.wait_obs(8000):   # 软件渲染下聚光灯点亮会触发一次光照着色器编译(秒级)，放宽超时兜住
                break
            meta = envh.read_meta()
            frame_ids.append(int(meta[0, E.M_FRAME]))
            steps_seq.append(int(meta[0, E.M_STEPS]))
            cross_env_var = max(cross_env_var, float(np.var(meta[:, E.M_STEPS])))
            envh.send_action(zeros_cont, zeros_disc)

        envh.close()
    finally:
        kill_godot(proc)
        log.close()

    # ---- 分析 ----
    ids = np.array(frame_ids)
    gaps = np.diff(ids)
    lossless = bool(np.all(gaps == 1))
    steps = np.array(steps_seq[1:])  # 跳过首帧

    print(f"    收到回合数            : {len(frame_ids)}")
    print(f"    [门控] 不发action时无新帧: {gated}   => {'PASS' if gated else 'FAIL'}")
    print(f"    [无丢] 帧号连续(+1)无跳号 : {lossless} (最大跳 {int(gaps.max()) if len(gaps) else 0})"
          f"   => {'PASS' if lossless else 'FAIL'}")
    print(f"    [一致] 跨环境步数最大方差 : {cross_env_var}   => {'PASS' if cross_env_var == 0 else 'FAIL'}")

    if mode.startswith("fix"):
        ok = bool(np.all(steps == fixed_steps))
        print(f"    [步数] 每帧恒为 X={fixed_steps} : {ok}  (实测唯一值 {sorted(set(steps.tolist()))})"
              f"   => {'PASS' if ok else 'FAIL'}")
    else:
        in_range = bool(steps.min() >= 1 and steps.max() <= max_steps)
        varies = bool(steps.min() != steps.max())
        print(f"    [步数] 落在[1,{max_steps}]且变化 : range=[{steps.min()},{steps.max()}] varies={varies}"
              f"   => {'PASS' if (in_range and varies) else 'FAIL'}")
        ok = in_range and varies
        vals, cnts = np.unique(steps, return_counts=True)
        print("    步数分布: " + ", ".join(f"{v}:{c}" for v, c in zip(vals.tolist(), cnts.tolist())))

    passed = gated and lossless and (cross_env_var == 0) and ok
    print(f"    => 本模式 {'[ PASS ]' if passed else '[ FAIL ]'}")
    return passed


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("=" * 70)
    print(" 步进模式 / action门控 / 无丢帧  验证")
    print("=" * 70)

    p1 = run_mode("模式1 固定步进 (Fixed, X=3)", "fixed", 3, 8,
                  os.path.join(E.PROJECT_DIR, "_godot_mode_fixed.log"))
    p2 = run_mode("模式2 任意步进 (Decoupled, Max=8)", "decoupled", 3, 8,
                  os.path.join(E.PROJECT_DIR, "_godot_mode_decoupled.log"))

    print()
    print("=" * 70)
    print(f"总判定: {'[ ALL PASS ]' if (p1 and p2) else '[ FAIL ]'}")
    return 0 if (p1 and p2) else 1


if __name__ == "__main__":
    sys.exit(main())
