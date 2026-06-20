"""
异步(自由跑)模式的【最小】Python 执行器。

异步模式由 Godot 侧完整实现(Main.cs ProcessAsync)：Godot 不等 Python，每 tick 读 SHM 最新动作→步进→
把 obs+累积reward 写回 SHM(seqlock 防撕裂)。Python 这里只做最小事：读最新 obs、写动作。

本脚本用来验证/测速：
  - Godot 发布率(frameCount 增长/秒) —— 应接近 Godot 满速(24步/帧 ~60/s)，证明不再被 Python 卡。
  - Python 消费率(本循环读到不同帧/秒) + 撕裂重试统计。

用法: python async_min.py [秒数=10]
"""

import os
import struct
import subprocess
import sys
import time

import numpy as np

import rl_train_env as E
import train_ppo as base


def read_seq(shm):
    shm.seek(E.SEQ_OFFSET)
    return struct.unpack("<i", shm.read(4))[0]


def read_consistent(env, retries=16):
    """seqlock 读：序号为偶且读前后一致才算无撕裂。总返回有效(可能撕裂)的 (imgs, meta, ok)。"""
    imgs = env.read_images()
    meta = env.read_meta()
    for _ in range(retries):
        s1 = read_seq(env.shm)
        imgs = env.read_images()
        meta = env.read_meta()
        s2 = read_seq(env.shm)
        if (s1 & 1) == 0 and s1 == s2:
            return imgs, meta, True
    return imgs, meta, False


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0

    run_env = os.environ.copy()
    run_env["RL_ASYNC"] = "1"
    run_env["RL_FIXED_STEPS"] = "24"
    log = open(os.path.join(base.PROJECT_DIR, "_async_min_godot.log"), "w",
               encoding="utf-8", errors="replace")
    proc = subprocess.Popen([base.GODOT_EXE, "--path", base.PROJECT_DIR, base.TRAIN_SCENE],
                            stdout=log, stderr=subprocess.STDOUT, env=run_env)
    try:
        env = E.GodotTrainEnv(connect_timeout_s=60)
        assert env.wait_obs(8000), "未收到首帧"
        zeros_c = np.zeros((E.NUM_ENVS, E.CONT_DIM), np.float32)

        reads = 0
        distinct = 0
        torn = 0
        last_fc = -1
        f0 = None
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < secs:
            imgs, meta, ok = read_consistent(env)
            if not ok:
                torn += 1
            fc = float(meta[0, E.M_FRAME])
            if f0 is None:
                f0 = fc
            if fc != last_fc:
                distinct += 1
                last_fc = fc
            reads += 1
            # 最小执行：写随机离散动作(前4槽=上/下/左/右)。Godot 自由读取最新值。
            disc = np.zeros((E.NUM_ENVS, E.DISC_DIM), np.int32)
            disc[:, :4] = (np.random.rand(E.NUM_ENVS, 4) < 0.3).astype(np.int32)
            env.send_action(zeros_c, disc)
        dt = time.perf_counter() - t0
        f1 = float(meta[0, E.M_FRAME])

        print("-" * 56)
        print(f"Godot 发布率(自由跑) : {(f1 - f0) / dt:.1f}/s  (frameCount {f0:.0f}->{f1:.0f})")
        print(f"Python 消费率(不同帧): {distinct / dt:.1f}/s")
        print(f"Python 读循环率      : {reads / dt:.1f}/s  ({reads} 次)")
        print(f"撕裂重试耗尽次数     : {torn}")
        env.close()
        return 0
    finally:
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            proc.kill()
        log.close()


if __name__ == "__main__":
    sys.exit(main())
