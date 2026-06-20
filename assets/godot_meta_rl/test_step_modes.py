"""
验证两种步进模式 + action 门控 + 无丢帧（同步 lock-step 协议）。

对每种模式启动一次 Godot，跑若干同步回合，校验：
  [门控] 不发 action 时，Godot 不产出新观测（证明"action 返回前绝不步进"）。
  [无丢] renderFrameCount 连续递增(+1)，无跳号 → Python 读到了每一帧。
  [步数] Fixed: 每帧物理步数恒为 X；Decoupled: 步数在 [1,Max] 间变化且最小>=1。
  [一致] 同一帧里 40 个环境的步数一致（方差 0）。
"""

import os
import subprocess
import sys
import time

import numpy as np

import rl_train_env as E

# 复用 rl_train_env 中的路径配置
GODOT_EXE = E.GODOT_EXE
PROJECT_DIR = E.PROJECT_DIR
TRAIN_SCENE = E.TRAIN_SCENE

CYCLES = 240


def launch(mode, fixed_steps, max_steps, log_path):
    env = os.environ.copy()
    env["RL_STEP_MODE"] = mode
    env["RL_FIXED_STEPS"] = str(fixed_steps)
    env["RL_MAX_STEPS"] = str(max_steps)
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen([GODOT_EXE, "--path", PROJECT_DIR, TRAIN_SCENE],
                            stdout=log, stderr=subprocess.STDOUT, env=env)
    return proc, log


def kill(proc):
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        proc.kill()


def run_mode(title, mode, fixed_steps, max_steps, log_path):
    print(f"\n### {title}  (RL_STEP_MODE={mode}, X={fixed_steps}, Max={max_steps})")
    proc, log = launch(mode, fixed_steps, max_steps, log_path)
    zeros_cont = np.zeros((E.NUM_ENVS, E.CONT_DIM), np.float32)
    zeros_disc = np.zeros((E.NUM_ENVS, E.DISC_DIM), np.int32)
    try:
        envh = E.GodotTrainEnv(connect_timeout_s=40)

        # 拿到第一帧观测（reset，steps=0）
        assert envh.wait_obs(5000), "未收到首帧观测"
        _ = envh.read_meta()

        # [门控] 不发 action，看 Godot 是否"卡住不产新帧"
        gated = not envh.wait_obs(500)   # 期望超时(无新帧)
        envh.send_action(zeros_cont, zeros_disc)  # 解除，进入正常回合

        frame_ids = []
        steps_seq = []
        cross_env_var = 0.0
        for _ in range(CYCLES):
            if not envh.wait_obs(2000):
                break
            meta = envh.read_meta()
            frame_ids.append(int(meta[0, E.M_FRAME]))
            steps_seq.append(int(meta[0, E.M_STEPS]))
            cross_env_var = max(cross_env_var, float(np.var(meta[:, E.M_STEPS])))
            envh.send_action(zeros_cont, zeros_disc)

        envh.close()
    finally:
        kill(proc)
        log.close()

    # ---- 分析 ----
    ids = np.array(frame_ids)
    gaps = np.diff(ids)
    lossless = bool(np.all(gaps == 1))
    steps = np.array(steps_seq[1:])  # 跳过 reset 帧后第一帧（可能仍是首个真实步）

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
        # 直方图
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
                  os.path.join(PROJECT_DIR, "_godot_mode_fixed.log"))
    p2 = run_mode("模式2 任意步进 (Decoupled, Max=8)", "decoupled", 3, 8,
                  os.path.join(PROJECT_DIR, "_godot_mode_decoupled.log"))

    print()
    print("=" * 70)
    print(f"总判定: {'[ ALL PASS ]' if (p1 and p2) else '[ FAIL ]'}")
    return 0 if (p1 and p2) else 1


if __name__ == "__main__":
    sys.exit(main())
