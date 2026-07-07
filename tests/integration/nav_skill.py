#!/usr/bin/env python3
"""导航技能终审:走到可见地标 N 格内(闭环成功率)。测 aim+attack 之外的**运动**模态。

远处放 glowstone 柱地标(带横向偏移→须转向),平坦地。成功=玩家水平距目标 < 2.5 格。
教师=按方位角闭环(读 xyz/yaw 特权算 bearing → 转向 + 前进)。策略 teacher/noop/learned(视觉快塔)。

用法(ZEROCOPY :1):
  DISPLAY=:1 PYTHONPATH=. ./.venv/bin/python tests/integration/nav_skill.py \
      --policy teacher --episodes 20 --steps 200 --save_demo runs/data/demo_navigate
"""
import argparse
import json
import math
import os

import numpy as np

from tests.integration.collect_s8 import DEG2PX, V2_KEYS, frame_pair
from tests.integration.skill_ceiling import LearnedFastHead, _np_rgb

VARIANTS = [(0, 12), (4, 12), (-4, 12), (6, 10), (-6, 10), (3, 15)]   # (横偏, 前距)
REACH = 3.0          # 地标是实心柱,玩家撞上停 ~2.7;到 3 格内=已导航到位


def build_nav_course(off_x, dist):
    """平坦地 + 正前方 dist、横偏 off_x 处竖一根 glowstone 地标(高 5,远处可见)。"""
    cmds = ["gamemode survival @p", "difficulty peaceful", "tp @p ~ ~ ~ 0 0",
            "fill ~-20 ~-1 ~-2 ~20 ~6 ~20 minecraft:air",
            "fill ~-20 ~-2 ~-2 ~20 ~-2 ~20 minecraft:stone", "clear @p"]
    for dy in range(5):
        cmds.append(f"setblock ~{off_x} ~{dy} ~{dist} minecraft:glowstone")
    return cmds


def dist_xz(full, tx, tz):
    return math.hypot(full.x - tx, full.z - tz)


class NavPolicy:
    """按方位角闭环:desired_yaw=atan2(-ox,oz)(MC 约定 yaw0=+Z),转向 + 前进。"""

    def __init__(self, rng):
        self.rng = rng
        self.tx = self.tz = None

    def set_target(self, full, off_x, dist):
        self.tx, self.tz = full.x + off_x, full.z + dist    # 出生点相对 → 世界坐标

    def __call__(self, t, noop, obs=None):
        a = dict(noop); full = obs["full"]
        ox, oz = self.tx - full.x, self.tz - full.z
        desired = math.degrees(math.atan2(-ox, oz))
        dyaw = (desired - float(full.yaw) + 180) % 360 - 180
        a["camera_yaw"] = float(np.clip(dyaw, -25, 25))
        a["forward"] = True                                 # 一直走(到位由 run 循环判成功)
        return a


def run(policy_name, episodes, steps, settle, port, seed, out, ckpt, max_len, greedy, save_demo):
    import torch
    from craftground import make
    from craftground.initial_environment_config import InitialEnvironmentConfig, WorldType
    from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
    from craftground.screen_encoding_modes import ScreenEncodingMode
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    zc = os.environ.get("DISPLAY", "") == ":1"
    env = make(initial_env_config=InitialEnvironmentConfig(
        image_width=640, image_height=360,
        screen_encoding_mode=ScreenEncodingMode.ZEROCOPY if zc else ScreenEncodingMode.RAW,
        world_type=WorldType.SUPERFLAT, seed="nav", request_raycast=True,
        initial_extra_commands=["gamemode survival @p"]),
        action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN, port=port, verbose=False)
    noop = no_op_v2(); env.reset(); rng = np.random.default_rng(seed)
    learned = LearnedFastHead(ckpt, dev, max_len, greedy, rng) if policy_name == "learned" else None
    rows = []
    for ep in range(episodes):
        off_x, dist = VARIANTS[ep % len(VARIANTS)]
        obs, _ = env.reset(options={"fast_reset": True, "extra_commands": build_nav_course(off_x, dist)})
        for _ in range(settle):
            obs, *_ = env.step(noop)
        tx, tz = obs["full"].x + off_x, obs["full"].z + dist
        if policy_name == "teacher":
            policy = NavPolicy(rng); policy.tx, policy.tz = tx, tz
        elif policy_name == "noop":
            policy = lambda t, noop, obs=None: dict(noop)
        else:
            policy = learned
        d0 = dist_xz(obs["full"], tx, tz)
        frames, dxs, dys, keys_l = [], [], [], []
        reached = False
        for t in range(steps):
            if save_demo:
                frames.append(frame_pair(_np_rgb(obs["rgb"]))[0])
            a = policy(t, noop, obs)
            if save_demo:
                dxs.append(float(a.get("camera_yaw", 0.0)) * DEG2PX)
                dys.append(float(a.get("camera_pitch", 0.0)) * DEG2PX)
                keys_l.append([float(bool(a.get(k, False))) for k in V2_KEYS])
            obs, *_ = env.step(a)
            if dist_xz(obs["full"], tx, tz) < REACH:
                reached = True; break
        dmin = dist_xz(obs["full"], tx, tz)
        if save_demo and frames and reached:
            frames.append(frame_pair(_np_rgb(obs["rgb"]))[0])
            os.makedirs(save_demo, exist_ok=True)
            np.savez_compressed(os.path.join(save_demo, f"navigate_ep{ep:03d}.npz"),
                                frames=np.stack(frames).astype(np.uint8),
                                dx=np.array(dxs, np.float32), dy=np.array(dys, np.float32),
                                keys=np.array(keys_l, np.uint8), gui=np.zeros(len(dxs), np.uint8),
                                score=np.float32(1), policy_strong=np.int64(1), start_hard=np.int64(0))
        rows.append({"ep": ep, "off_x": off_x, "dist": dist, "reached": reached,
                     "d0": round(d0, 2), "dmin": round(dmin, 2)})
        print(f"[nav/{policy_name}] ep{ep} off={off_x} dist={dist} reached={reached} d0={d0:.1f}→{dmin:.1f}", flush=True)
    env.close()
    rate = float(np.mean([r["reached"] for r in rows]))
    res = {"policy": policy_name, "episodes": episodes, "success_rate": round(rate, 4),
           "per_episode": rows}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(res, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"[nav/{policy_name}] SUCCESS_RATE={rate:.3f} → {out}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="teacher")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--greedy", action="store_true", default=False)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--settle", type=int, default=10)
    p.add_argument("--port", type=int, default=9100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_demo", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = args.out or f"runs/ceiling/navigate_{args.policy}.json"
    run(args.policy, args.episodes, args.steps, args.settle, args.port, args.seed, out,
        args.ckpt, args.max_len, args.greedy, args.save_demo)


if __name__ == "__main__":
    main()
