#!/usr/bin/env python3
"""木类 GT 采集(引擎 B 扩类;设计见 net/fovea_twotower/wood.py 档头)。

每局:spreadplayers 随机落点 → raycast 网格扫掠(tp 指定 yaw/pitch,读准星
方块)累积 log 方块坐标 → 命中≥min_logs 则录采集帧(观察策略,不攻击),
gt={"log": 命中集};零命中则录"认证无树"负帧(gt 全空)。

用法:
  DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1 CRAFTGROUND_JVM_MAX_MEMORY=2G PYTHONPATH=. \
    .venv/bin/python tests/integration/collect_wood_gt.py --episodes 30 \
      --out_pos runs/data/wood_gt --out_neg runs/data/wood_negcert --port 8820
"""
import argparse
import json
import os
import time

import numpy as np

from net.fovea_twotower.wood import WOOD_CLASSES
from tests.integration.collect_calib640 import (ObservePolicy, _frame, _pose,
                                                _ray)
from tests.integration.collect_calib_natural import relocate_cmds

SCAN_PITCHES = (42, 20, 4, -14, -30)            # 近距下段树干到树冠


def scan_logs(env, noop, rate=15.0):
    """相机连续扫掠(增量动作,tp 置向不生效——相机权威在客户端鼠标状态):
    每个 pitch 带先闭环对准 pitch,再以 rate°/step 转满一圈,逐步读 raycast。"""
    found = {}
    obs, *_ = env.step(noop)
    for tgt_pitch in SCAN_PITCHES:
        for _ in range(8):                       # 闭环对准 pitch
            cur = float(getattr(obs["full"], "pitch", 0.0))
            dp = tgt_pitch - cur
            if abs(dp) < 2:
                break
            a = dict(noop)
            a["camera_pitch"] = float(np.clip(dp, -12, 12))
            obs, *_ = env.step(a)
        for _ in range(int(360 / rate) + 2):     # 转满一圈
            a = dict(noop)
            a["camera_yaw"] = rate
            obs, *_ = env.step(a)
            xyz, key, d = _ray(obs["full"])
            if "log" in key and 0 < d < 24:
                found[tuple(xyz)] = True
    cols = {}
    for (x, y, z) in found:                      # 同列竖直补洞:树干=竖列
        cols.setdefault((x, z), []).append(y)
    full = []
    for (x, z), ys in cols.items():
        for y in range(min(ys), max(ys) + 1):
            full.append([x, y, z])
    return full, obs


def run(args):
    from craftground import make
    from craftground.initial_environment_config import (InitialEnvironmentConfig,
                                                        WorldType)
    from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
    from craftground.screen_encoding_modes import ScreenEncodingMode

    os.makedirs(args.out_pos, exist_ok=True)
    os.makedirs(args.out_neg, exist_ok=True)
    cfg = InitialEnvironmentConfig(
        image_width=640, image_height=360,
        screen_encoding_mode=ScreenEncodingMode.RAW,
        world_type=WorldType.DEFAULT, seed=args.world_seed,
        request_raycast=True,
        initial_extra_commands=["gamemode survival @p", "difficulty peaceful"])
    env = make(initial_env_config=cfg,
               action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
               port=args.port, verbose=False)
    noop = no_op_v2()
    env.reset()
    print(f"[wood] env up (port {args.port})", flush=True)
    rng = np.random.default_rng(args.seed)
    n_pos = n_neg = 0
    for ep in range(args.episodes):
        t0 = time.time()
        obs, _ = env.reset(options={"fast_reset": True,
                                    "extra_commands": relocate_cmds(rng)})
        for _ in range(args.settle):
            obs, *_ = env.step(noop)
        time.sleep(2.5)
        for _ in range(5):
            obs, *_ = env.step(noop)
        logs, obs = scan_logs(env, noop)
        if 0 < len(logs) < args.min_logs:      # 弱命中→朝最近树走近→复扫
            pose = _pose(obs["full"])
            b = min(logs, key=lambda q: (q[0]-pose[0])**2 + (q[2]-pose[2])**2)
            tgt_yaw = float(np.degrees(np.arctan2(-(b[0]+.5-pose[0]),
                                                  b[2]+.5-pose[2])))
            for _ in range(30):
                cur = float(getattr(obs["full"], "yaw", 0.0))
                dy = (tgt_yaw - cur + 180) % 360 - 180
                a = dict(noop)
                a["camera_yaw"] = float(np.clip(dy, -15, 15))
                a["forward"] = abs(dy) < 25
                a["jump"] = True
                obs, *_ = env.step(a)
            logs2, obs = scan_logs(env, noop)
            logs = [list(k) for k in {tuple(q) for q in logs + logs2}]
        is_pos = len(logs) >= args.min_logs
        gt = {c: [] for c in WOOD_CLASSES}
        gt["log"] = logs
        pol = ObservePolicy(rng)
        frames, poses, rk = [], [], []
        for t in range(args.steps):
            a = pol(t, noop, obs)
            if is_pos and (t // 8) % 2 == 0 and logs:   # 半程视角偏向树
                pose_now = _pose(obs["full"])
                b = logs[rng.integers(len(logs))]
                ty = float(np.degrees(np.arctan2(-(b[0]+.5-pose_now[0]),
                                                 b[2]+.5-pose_now[2])))
                dy = (ty - pose_now[3] + 180) % 360 - 180
                a["camera_yaw"] = float(np.clip(dy, -14, 14))
            obs, *_ = env.step(a)
            if t % args.stride == 0:
                frames.append(_frame(obs["rgb"]))
                poses.append(_pose(obs["full"]))
                rk.append(_ray(obs["full"])[1])
        out_dir = args.out_pos if is_pos else args.out_neg
        tag = "pos" if is_pos else "negcert"
        outp = os.path.join(out_dir, f"wood_{tag}_s{args.seed}_e{ep}.npz")
        np.savez_compressed(
            outp, frames=np.stack(frames).astype(np.uint8),
            pose=np.array(poses, np.float32), ray_key=np.array(rk),
            gt_blocks=json.dumps(gt),
            meta=json.dumps({"n_logs": len(logs), "episode": ep}))
        if is_pos:
            n_pos += 1
        else:
            n_neg += 1
        print(f"[wood] ✓ e{ep} {tag} logs={len(logs)} "
              f"{time.time()-t0:.0f}s [pos {n_pos}/neg {n_neg}]", flush=True)
    env.close()
    print(f"[wood] DONE pos={n_pos} neg={n_neg}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_pos", default="runs/data/wood_gt")
    p.add_argument("--out_neg", default="runs/data/wood_negcert")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--settle", type=int, default=25)
    p.add_argument("--min_logs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--world_seed", default="woodgt1")
    p.add_argument("--port", type=int, default=8820)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
