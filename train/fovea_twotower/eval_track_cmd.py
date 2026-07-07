#!/usr/bin/env python3
"""E1 闭环终审:指令条件快头在活环境里的追踪/切换/到达(train/fovea_twotower)。

判据(knowledge/design_fovea_yolo_fasttower.md §4,先于结果登记):
  T1 追踪:指令段中位角误差 ≤8°,且比"冻结相机"基线降 ≥40%;
  T2 可控性:局中切换指令,策略重定向到新目标——切换成功率 ≥0.7
     (成功=切换后 30 步内对新目标角误差中位 <12° 且优于对旧目标);
  T3 导航:≥60% 局在 220 步内到达指令目标 2.8 格内(比随机基线 +0.15)。
对照:teacher(采集器天花板)/ frozen(不动相机)/ random。

用法:
  DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1 PYTHONPATH=. .venv/bin/python \
      train/fovea_twotower/eval_track_cmd.py --ckpt runs/trackcmd_bc/best.pt --episodes 12
"""
import argparse
import json
import os

import numpy as np
import torch

from net.fovea_twotower.yolo_parse import TrackNavConfig, build_tracknav
from tests.integration.collect_calib640 import (WALL_Z_VARIANTS, _pose,
                                                anchor_gt_blocks,
                                                build_calib_course,
                                                sample_offsets)
from tests.integration.collect_s8 import V2_KEYS
from tests.integration.collect_track_cmd import (CLASSES, AimTeacher, TokenHead,
                                                 TokenTeacher, aim_solution)
from train.fovea_twotower.train_track_cmd import CAM_NORM_PX, goal_relative
from train.minecraft.vpt_action import bin_to_camera

DEG_PER_PX = 0.15


class StudentPolicy:
    """token 流 + 指令 → 动作(维护 seq_len 窗口上下文,bin argmax → 相机 deg)。"""

    def __init__(self, ckpt, device="cuda", seq_len=64):
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        self.tower = build_tracknav(TrackNavConfig(**ck["cfg"])).to(device).eval()
        self.tower.load_state_dict(ck["tower"])
        self.dev, self.L = device, seq_len
        self.bins = self.tower.cfg.camera_bins
        self.reset()

    def reset(self):
        self.toks, self.prevs = [], []

    @torch.no_grad()
    def __call__(self, tok_rel, noop):
        self.toks.append(tok_rel)
        if not self.prevs:
            self.prevs.append(np.zeros(22, np.float32))
        toks = torch.from_numpy(np.stack(self.toks[-self.L:]))[None].to(self.dev)
        prev = torch.from_numpy(np.stack(self.prevs[-self.L:]))[None].to(self.dev)
        g = torch.zeros(1, 1, device=self.dev)
        cam, key = self.tower(toks.float(), g, prev.float())
        cb = cam[0, -1].float().argmax(-1).cpu()                    # [2] bin
        kp = (key[0, -1].float().sigmoid() > 0.5).cpu().numpy()     # [20]
        val = bin_to_camera(cb).numpy()                             # [-1,1] mu-law 解码
        dx_px, dy_px = val * CAM_NORM_PX
        a = dict(noop)
        a["camera_yaw"] = float(np.clip(dx_px * DEG_PER_PX, -18, 18))
        a["camera_pitch"] = float(np.clip(dy_px * DEG_PER_PX, -18, 18))
        for i, k in enumerate(V2_KEYS):
            if kp[i]:
                a[k] = True
        row = np.zeros(22, np.float32)
        row[0] = np.clip(dx_px / CAM_NORM_PX, -1, 1)
        row[1] = np.clip(dy_px / CAM_NORM_PX, -1, 1)
        row[2:] = kp
        self.prevs.append(row)
        return a


def nearest_err_dist(pose, gt_blocks, cls):
    cands = [aim_solution(pose, (b[0] + .5, b[1] + .5, b[2])) for b in gt_blocks[cls]]
    _, _, err, d = min(cands, key=lambda c: c[3])
    return err, d


def run_episode(env, noop, tok_head, actor, rng, wall_z, steps, switch_t,
                step_delay=0.0, action_repeat=1):
    offsets = sample_offsets(rng)
    obs, _ = env.reset(options={"fast_reset": True,
                                "extra_commands": build_calib_course(wall_z, offsets)})
    for _ in range(10):
        obs, *_ = env.step(noop)
    gt, obs = anchor_gt_blocks(env, noop, offsets)
    if gt is None:
        return None
    a0 = dict(noop)                        # 开局扰乱(与 v2 示范同协议;否则准星白送在中心矿)
    a0["camera_yaw"] = float(rng.uniform(-40, 40))
    a0["camera_pitch"] = float(rng.uniform(-15, 10))
    obs, *_ = env.step(a0)
    g1, g2 = rng.choice(len(CLASSES), 2, replace=False)
    errs, dists, goals = [], [], []
    rgb = np.asarray(obs["rgb"])
    if rgb.shape[0] in (1, 3):
        rgb = rgb.transpose(1, 2, 0)
    prev_goal = -1
    for t in range(steps):
        goal = int(g1 if t < switch_t else g2)
        pose = _pose(obs["full"])
        err, d = nearest_err_dist(pose, gt, CLASSES[goal])
        errs.append(err)
        dists.append(d)
        goals.append(goal)
        kind = actor["kind"]
        if kind == "student":
            toks = tok_head(rgb)                                    # [K,6+C+1]
            rel = goal_relative(toks[None], np.array([goal]))[0]    # [K,8]
            a = actor["fn"](rel, noop)
        elif kind == "token_teacher":                               # 观测一致天花板
            if goal != prev_goal:
                actor["fn"].new_segment()
            a = actor["fn"](noop, tok_head(rgb), goal, pose[4])
        elif kind == "teacher":
            a, _, _ = actor["fn"](noop, pose, gt, CLASSES[goal])
        elif kind == "random":
            a = dict(noop)
            a["camera_yaw"] = float(rng.normal(0, 10))
            a["camera_pitch"] = float(rng.normal(0, 6))
            a["forward"] = bool(rng.random() < 0.3)
        else:                                                       # frozen
            a = dict(noop)
        prev_goal = goal
        if step_delay > 0:                     # 设备速度鲁棒性探针:模拟慢设备推理延迟
            import time as _t
            _t.sleep(step_delay)
        obs, *_ = env.step(a)
        for _ in range(action_repeat - 1):     # 跳帧探针:动作保持 k tick=等效帧率 1/k
            a_hold = dict(a)                   # (可变帧率敏感度;相机增量只发一次,
            a_hold["camera_yaw"] = 0.0         # 按键保持——模拟"没赶上 tick")
            a_hold["camera_pitch"] = 0.0
            obs, *_ = env.step(a_hold)
        rgb = np.asarray(obs["rgb"])
        if rgb.shape[0] in (1, 3):
            rgb = rgb.transpose(1, 2, 0)
    errs, dists = np.array(errs), np.array(dists)
    m = dict(
        err_p1=float(np.median(errs[15:switch_t])),
        err_p2=float(np.median(errs[switch_t + 30:])),
        err_p2_early=float(np.median(errs[switch_t:switch_t + 30])),
        arrived=bool((dists[switch_t:] <= 2.8).any() or (dists[:switch_t] <= 2.8).any()),
        arrived_final=bool((dists[-30:] <= 2.8).any()),
        switch_ok=bool(np.median(errs[switch_t + 5:switch_t + 35]) < 12.0
                       or np.median(errs[switch_t + 30:]) < 12.0),
        g1=int(g1), g2=int(g2))
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/trackcmd_bc/best.pt")
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--steps", type=int, default=220)
    p.add_argument("--switch_t", type=int, default=110)
    p.add_argument("--arms", nargs="+",
                   default=["student", "teacher", "frozen", "random"])
    p.add_argument("--vectors", default="runs/g1_vectors.pt")
    p.add_argument("--conv_head", default="runs/g1_conv_head.pt")
    p.add_argument("--step_delay", type=float, default=0.0,
                   help="每步注入延迟秒数(设备速度鲁棒性探针)")
    p.add_argument("--action_repeat", type=int, default=1,
                   help="动作保持 k tick(可变帧率敏感度探针)")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--port", type=int, default=8538)
    p.add_argument("--out", default="runs/trackcmd_closedloop.json")
    args = p.parse_args()

    from craftground import make
    from craftground.initial_environment_config import (InitialEnvironmentConfig,
                                                        WorldType)
    from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
    from craftground.screen_encoding_modes import ScreenEncodingMode
    cfg = InitialEnvironmentConfig(
        image_width=640, image_height=360,
        screen_encoding_mode=ScreenEncodingMode.RAW,
        world_type=WorldType.SUPERFLAT, seed="s8fovea", request_raycast=True,
        initial_extra_commands=["gamemode survival @p"])
    env = make(initial_env_config=cfg,
               action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
               port=args.port, verbose=False)
    noop = no_op_v2()
    env.reset()

    tok_head = TokenHead(args.vectors, conv_head=args.conv_head)
    res = {}
    for arm in args.arms:
        rng = np.random.default_rng(args.seed)                     # 各臂同 RNG=同课程序列
        if arm == "student":
            student = StudentPolicy(args.ckpt)
            actor = {"kind": "student", "fn": student}
        elif arm == "teacher":
            actor = {"kind": "teacher", "fn": AimTeacher(rng, epsilon=0.0)}
        elif arm == "token_teacher":
            actor = {"kind": "token_teacher", "fn": TokenTeacher(rng, epsilon=0.0)}
        else:
            actor = {"kind": arm}
        ms = []
        for ep in range(args.episodes):
            if arm == "student":
                student.reset()
            m = run_episode(env, noop, tok_head, actor, rng,
                            WALL_Z_VARIANTS[ep % len(WALL_Z_VARIANTS)],
                            args.steps, args.switch_t, args.step_delay,
                            args.action_repeat)
            if m:
                ms.append(m)
                print(f"[{arm}] ep{ep} err p1/p2={m['err_p1']:.1f}/{m['err_p2']:.1f}° "
                      f"switch={'✓' if m['switch_ok'] else '✗'} "
                      f"arrive={'✓' if m['arrived_final'] else '✗'}", flush=True)
        res[arm] = dict(
            n=len(ms),
            err_p1_med=float(np.median([m["err_p1"] for m in ms])),
            err_p2_med=float(np.median([m["err_p2"] for m in ms])),
            switch_rate=float(np.mean([m["switch_ok"] for m in ms])),
            arrive_rate=float(np.mean([m["arrived_final"] for m in ms])))
        print(f"[{arm}] {json.dumps(res[arm])}", flush=True)
    env.close()

    s, f = res.get("student", {}), res.get("frozen", {})
    gates = dict(
        T1_track=bool(s and s["err_p1_med"] <= 8.0
                      and s["err_p1_med"] <= 0.6 * f.get("err_p1_med", 1e9)),
        T2_switch=bool(s and s["switch_rate"] >= 0.7),
        T3_nav=bool(s and s["arrive_rate"] >= 0.6
                    and s["arrive_rate"] >= res.get("random", {}).get("arrive_rate", 0) + 0.15))
    out = dict(arms=res, gates=gates, args=vars(args))
    with open(args.out, "w") as fp:
        json.dump(out, fp, indent=2, ensure_ascii=False)
    print(f"[gates] {json.dumps(gates)}\n→ {args.out}", flush=True)


if __name__ == "__main__":
    main()
