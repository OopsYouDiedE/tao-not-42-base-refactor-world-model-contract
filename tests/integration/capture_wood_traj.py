#!/usr/bin/env python3
"""抓一局快塔砍树轨迹并把感知/动作叠到帧上(诊断可视化,非训练)。

每帧叠:快塔看见的 log token 框(绿,goal 目标加粗)+ 动作(相机箭头/attack/forward)
+ 状态条(tick / 看见log / 挖掘闩锁 / raycast命中)。输出标注帧序列 npz,
供 build_wood_traj_artifact 拼成可播放页面。

用法:
  DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1 CRAFTGROUND_JVM_MAX_MEMORY=2G PYTHONPATH=. \
    .venv/bin/python tests/integration/capture_wood_traj.py --port 8865 --max_steps 400
"""
import argparse
import time

import cv2
import numpy as np

from net.fovea_twotower.token_stream import TokenHead, as_hwc, goal_relative
from net.fovea_twotower.wood import WOOD_CLASSES
from tests.integration.collect_calib640 import _ray
from tests.integration.collect_calib_natural import relocate_cmds
from tests.integration.fullloop_chain import env_inventory
from tests.integration.wood_chain import place_trees


def annotate(rgb, toks, gcls, a, saw, latch, dist, t):
    """rgb[360,640,3] + token/动作/状态 → 叠加 BGR 帧。"""
    im = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    H, W = im.shape[:2]
    # log token 框(几何在 384×640 pad 系,cx,cy 归一 → 映射回 360)
    for k in range(len(toks)):
        p_log = float(toks[k, 6 + gcls])
        if toks[k, 4] <= 0 or p_log < 0.25:
            continue
        cx, cy, w, h = toks[k, 0] * W, toks[k, 1] * H, toks[k, 2] * W, toks[k, 3] * H
        x1, y1 = int(cx - w / 2), int(cy - h / 2)
        x2, y2 = int(cx + w / 2), int(cy + h / 2)
        thick = 3 if p_log > 0.4 else 1
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), thick)
        cv2.putText(im, f"log {p_log:.2f}", (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    # 相机动作箭头(中心)
    cxc, cyc = W // 2, H // 2
    dx = int(a.get("camera_yaw", 0) * 4)
    dy = int(a.get("camera_pitch", 0) * 4)
    if abs(dx) + abs(dy) > 2:
        cv2.arrowedLine(im, (cxc, cyc), (cxc + dx, cyc + dy), (0, 200, 255), 2, tipLength=0.3)
    # 状态条
    bar = [f"t={t}"]
    if a.get("attack"):
        bar.append("ATTACK")
    if a.get("forward"):
        bar.append("FWD")
    bar.append("see_log" if saw else "no_log")
    if latch:
        bar.append("MINE-LATCH")
    if dist and dist > 0:
        bar.append(f"ray={dist:.1f}")
    cv2.rectangle(im, (0, 0), (W, 20), (0, 0, 0), -1)
    cv2.putText(im, " | ".join(bar), (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1)
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/trackcmd_bc_v17/best.pt")
    p.add_argument("--conv_head", default="runs/g1_conv_head_v7b_wood.pt")
    p.add_argument("--vectors", default="runs/g1_vectors.pt")
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--save_every", type=int, default=2)
    p.add_argument("--port", type=int, default=8865)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="runs/wood_traj_capture.npz")
    args = p.parse_args()

    from craftground import make
    from craftground.initial_environment_config import (InitialEnvironmentConfig,
                                                        WorldType)
    from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
    from craftground.screen_encoding_modes import ScreenEncodingMode
    from train.fovea_twotower.eval_track_cmd import StudentPolicy

    tok_head = TokenHead(args.vectors, conv_head=args.conv_head, classes=WOOD_CLASSES)
    student = StudentPolicy(args.ckpt)
    gcls = WOOD_CLASSES.index("log")
    rng = np.random.default_rng(args.seed)
    cfg = InitialEnvironmentConfig(
        image_width=640, image_height=360, screen_encoding_mode=ScreenEncodingMode.RAW,
        world_type=WorldType.DEFAULT, seed="woodtraj", request_raycast=True,
        initial_extra_commands=["gamemode survival @p", "difficulty peaceful"])
    env = make(initial_env_config=cfg,
               action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
               port=args.port, verbose=False)
    import atexit
    _done = []

    def _sd():
        if _done:
            return
        _done.append(1)
        try:
            env.close()
        except Exception:
            pass
    atexit.register(_sd)
    noop = no_op_v2()
    env.reset()
    obs, _ = env.reset(options={"fast_reset": True, "extra_commands": relocate_cmds(rng)})
    for _ in range(20):
        obs, *_ = env.step(noop)
    time.sleep(2.0)
    obs = place_trees(env, noop, rng)
    student.reset()
    rgb = as_hwc(obs["rgb"])
    frames, saw_n, latch_n = [], 0, 0
    for t in range(args.max_steps):
        _xyz, key, dist = _ray(obs["full"])
        latch = "log" in key and 0 < dist <= 4.5
        toks = tok_head(rgb)
        saw = bool(len(toks) and float(toks[:, 6 + gcls].max()) > 0.4)
        saw_n += saw
        if latch:
            latch_n += 1
            a = dict(noop)
            a["attack"] = True
            if t % 12 == 0:
                a["forward"] = True
        else:
            rel = goal_relative(toks[None], np.array([gcls]))[0]
            a = student(rel, noop)
        if t % args.save_every == 0:
            frames.append(annotate(rgb, toks, gcls, a, saw, latch, dist, t))
        obs, *_ = env.step(a)
        rgb = as_hwc(obs["rgb"])
        if t % 10 == 0 and any("log" in i for i in env_inventory(obs["full"])):
            frames.append(annotate(rgb, tok_head(rgb), gcls, dict(noop), saw, latch, dist, t))
            break
    _sd()
    got = any("log" in i for i in env_inventory(obs["full"]))
    np.savez_compressed(args.out, frames=np.array(frames, np.uint8),
                        saw=saw_n, latch=latch_n, steps=t + 1, got_log=got)
    print(f"[capture] {len(frames)}帧 saw={saw_n} latch={latch_n} got_log={got} → {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
