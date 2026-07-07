#!/usr/bin/env python3
"""A1 全系统组装联审:YOLOE 前端 + 快头(F1 胜者) + 北锚定地图 + 慢塔查询,真实 MC。

M3 全部结论来自合成世界——A1 是地图模块第一次接真实感知/真实里程计:
  感知→图:TokenHead 检测 → 方位(cx→角度+yaw 本体感受) + bbox 面积标定距离;
  里程计:动作积分(步速=一次性标定,本体感受),碰撞=真实漂移源
          (顶墙按 forward 时里程计照记、真实位移为零——MC 典型误差,免注入);
  自定位:EgoMapNorthLoc 互相关修正(e3_fine 配方:0.5 格分辨率/每 3 步);
  慢塔口:MapQuery.nearest(目标类) 文本 + 回家向量 -p̂。
GT(obs.full 位置)只做打分,策略/地图链路零特权。

标定纪律:步速/面积-距离/方位符号 在标定局(GT 允许,传感器标定性质)拟合,
评估局冻结。判据先登记(docs/architectures/fovea-experiments-index.md A1 表)。

用法:
  DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1 PYTHONPATH=. .venv/bin/python \
      tests/integration/assembly_a1.py --ckpt <F1胜者> --episodes 12 --port 8720
"""
import argparse
import json

import numpy as np
import torch

from net.fovea_twotower.ego_map import EgoMapNorthLoc, MapQuery, _bearing_cn
from net.fovea_twotower.token_stream import CLASSES, TokenHead, as_hwc, goal_relative
from tests.integration.collect_calib640 import (WALL_Z_VARIANTS, _pose,
                                                anchor_gt_blocks,
                                                build_calib_course,
                                                sample_offsets)
from train.fovea_twotower.eval_track_cmd import StudentPolicy

DEG_PER_PX = 0.15
C = len(CLASSES)
RELOC_CFG = dict(window=6, min_ratio=1.15, subcell=False, min_pts=1.0)


def yaw_dir(yaw_deg):
    """MC 朝向单位向量(x,z):yaw0=+z(南),+yaw 向 -x。"""
    r = np.radians(yaw_deg)
    return np.array([-np.sin(r), np.cos(r)])


def det_world_xz(tok, pose, area_k, brg_sign):
    """token → 世界系相对 (x,z):方位=yaw+sign·像素偏角,距离=k/√面积。"""
    cx_px, area = tok[0] * 640, max(tok[4] * 640 * 384, 1.0)
    brg = pose[3] + brg_sign * (cx_px - 320) * DEG_PER_PX
    d = area_k / np.sqrt(area)
    return yaw_dir(brg) * d


def calibrate(env, noop, tok_head, rng, wall_z, steps=120):
    """标定局(GT 允许):步速 + 面积-距离 k + 方位符号。"""
    offsets = sample_offsets(rng)
    obs, _ = env.reset(options={"fast_reset": True,
                                "extra_commands": build_calib_course(wall_z, offsets)})
    for _ in range(10):
        obs, *_ = env.step(noop)
    gt, obs = anchor_gt_blocks(env, noop, offsets)
    if gt is None:
        return None
    speeds, pairs = [], []          # pairs: (area_px, gt_d, cx_px, gt_brg_rel_deg)
    prev_xz = None
    for t in range(steps):
        pose = _pose(obs["full"])
        xz = np.array([pose[0], pose[2]])
        a = dict(noop)
        a["camera_yaw"] = float(rng.normal(0, 8))
        a["forward"] = True
        if prev_xz is not None:
            speeds.append(float(np.linalg.norm(xz - prev_xz)))
        prev_xz = xz
        toks = tok_head(as_hwc(obs["rgb"]))
        for tok in toks:
            ci = int(np.argmax(tok[6:6 + C]))
            if tok[6 + ci] < 0.4 or tok[4] <= 0:
                continue
            rels = [np.array([b[0] + .5 - pose[0], b[2] + .5 - pose[2]])
                    for b in gt[CLASSES[ci]]]
            if not rels:
                continue
            rel = min(rels, key=np.linalg.norm)
            gt_d = float(np.linalg.norm(rel))
            gt_brg = np.degrees(np.arctan2(-rel[0], rel[1]))     # MC yaw 系
            rel_brg = (gt_brg - pose[3] + 180) % 360 - 180
            if abs(rel_brg) < 50 and gt_d < 25:
                pairs.append((tok[4] * 640 * 384, gt_d, tok[0] * 640, rel_brg))
        obs, *_ = env.step(a)
    v = float(np.median([s for s in speeds if s > 0.02]))       # 碰撞步剔除
    ar = np.array(pairs)
    k = float(np.median(ar[:, 1] * np.sqrt(ar[:, 0])))
    err_p = np.median(np.abs(ar[:, 3] - (ar[:, 2] - 320) * DEG_PER_PX))
    err_m = np.median(np.abs(ar[:, 3] + (ar[:, 2] - 320) * DEG_PER_PX))
    sign = 1.0 if err_p <= err_m else -1.0
    return dict(v=v, area_k=k, brg_sign=sign, n_pairs=len(pairs),
                brg_resid=float(min(err_p, err_m)))


def run_episode(env, noop, tok_head, student, rng, wall_z, cal,
                steps=220, switch_t=110, every=3):
    offsets = sample_offsets(rng)
    obs, _ = env.reset(options={"fast_reset": True,
                                "extra_commands": build_calib_course(wall_z, offsets)})
    for _ in range(10):
        obs, *_ = env.step(noop)
    gt, obs = anchor_gt_blocks(env, noop, offsets)
    if gt is None:
        return None
    a0 = dict(noop)
    a0["camera_yaw"] = float(rng.uniform(-40, 40))
    a0["camera_pitch"] = float(rng.uniform(-15, 10))
    obs, *_ = env.step(a0)
    student.reset()
    g1, g2 = rng.choice(C, 2, replace=False)
    pose0 = _pose(obs["full"])
    origin = np.array([pose0[0], pose0[2]])
    m = EgoMapNorthLoc(C, 128, 32.0)
    p_dead, p_reloc = np.zeros(2), np.zeros(2)
    e_dead, e_reloc = [], []
    prev_fwd_yaw = None
    for t in range(steps):
        goal = int(g1 if t < switch_t else g2)
        pose = _pose(obs["full"])
        p_gt = np.array([pose[0], pose[2]]) - origin
        # —— 里程计:上一步按下 forward 则记 v̂·dir(yaw);碰撞漂移天然存在 ——
        if prev_fwd_yaw is not None:
            d_odo = cal["v"] * yaw_dir(prev_fwd_yaw)
            p_dead += d_odo
            p_reloc += d_odo
            m.step(d_odo)
        toks = tok_head(as_hwc(obs["rgb"]))
        pts, fts = [], []
        for tok in toks:
            ci = int(np.argmax(tok[6:6 + C]))
            if tok[6 + ci] < 0.4 or tok[4] <= 0:
                continue
            pts.append(det_world_xz(tok, pose, cal["area_k"], cal["brg_sign"]))
            fts.append(np.eye(C)[ci])
        if pts:
            pts_t = torch.from_numpy(np.array(pts)).float()
            fts_t = torch.from_numpy(np.array(fts)).float()
            if t % every == 0 and t > 10:
                e_hat = m.relocalize(pts_t, fts_t, **RELOC_CFG)
                if e_hat is not None:
                    p_reloc += e_hat
                    m.step(e_hat)
            m.write(pts_t, fts_t)
        e_dead.append(float(np.linalg.norm(p_dead - p_gt)))
        e_reloc.append(float(np.linalg.norm(p_reloc - p_gt)))
        rel = goal_relative(toks[None], np.array([goal]))[0]
        a = student(rel, noop)
        prev_fwd_yaw = pose[3] if a.get("forward") else None
        obs, *_ = env.step(a)
    # —— 期末考:回家向量 + 慢塔查询(目标类最近实例方位) ——
    pose = _pose(obs["full"])
    p_gt = np.array([pose[0], pose[2]]) - origin
    def brg_ok(v_est, v_gt, tol=45.0):
        if np.linalg.norm(v_est) < 1e-6 or np.linalg.norm(v_gt) < 1e-6:
            return None
        d = np.degrees(np.arccos(np.clip(np.dot(v_est, v_gt) /
                                         (np.linalg.norm(v_est) * np.linalg.norm(v_gt)),
                                         -1, 1)))
        return bool(d <= tol)
    home_ok = brg_ok(-p_reloc, -p_gt)
    rels = [np.array([b[0] + .5 - pose[0], b[2] + .5 - pose[2]])
            for b in gt[CLASSES[int(g2)]]]
    q_ok = None
    if rels:
        gt_rel = min(rels, key=np.linalg.norm)
        v, txt = MapQuery(m, CLASSES).nearest(CLASSES[int(g2)])
        if v is not None:
            vv = np.array([float(v[0]), float(v[1])]) - p_reloc  # 图系→自身相对
            q_ok = brg_ok(vv, gt_rel)
    return dict(dead_end=e_dead[-1], reloc_end=e_reloc[-1],
                dead_med=float(np.median(e_dead[30:])),
                reloc_med=float(np.median(e_reloc[30:])),
                home_ok=home_ok, query_ok=q_ok)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--steps", type=int, default=220)
    p.add_argument("--switch_t", type=int, default=110)
    p.add_argument("--vectors", default="runs/g1_vectors.pt")
    p.add_argument("--conv_head", default="runs/g1_conv_head_v4.pt")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--port", type=int, default=8720)
    p.add_argument("--out", default="runs/assembly_a1.json")
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
    rng = np.random.default_rng(args.seed)
    cal = calibrate(env, noop, tok_head, rng, WALL_Z_VARIANTS[0])
    print(f"[calib] {json.dumps(cal)}", flush=True)
    student = StudentPolicy(args.ckpt)
    ms = []
    for ep in range(args.episodes):
        r = run_episode(env, noop, tok_head, student, rng,
                        WALL_Z_VARIANTS[ep % len(WALL_Z_VARIANTS)],
                        cal, args.steps, args.switch_t)
        if r:
            ms.append(r)
            print(f"[a1] ep{ep} dead={r['dead_end']:.2f} reloc={r['reloc_end']:.2f} "
                  f"home={'✓' if r['home_ok'] else '✗' if r['home_ok'] is not None else '-'} "
                  f"query={'✓' if r['query_ok'] else '✗' if r['query_ok'] is not None else '-'}",
                  flush=True)
    env.close()
    dead = float(np.median([r["dead_end"] for r in ms]))
    reloc = float(np.median([r["reloc_end"] for r in ms]))
    homes = [r["home_ok"] for r in ms if r["home_ok"] is not None]
    qs = [r["query_ok"] for r in ms if r["query_ok"] is not None]
    out = dict(n=len(ms), dead_end_med=round(dead, 2), reloc_end_med=round(reloc, 2),
               home_rate=round(float(np.mean(homes)), 2) if homes else None,
               query_rate=round(float(np.mean(qs)), 2) if qs else None,
               calib=cal,
               gates={
                   "G-A1(reloc<=dead)": bool(reloc <= dead),
                   "G-A2(home方位45°>=0.7)": bool(homes and np.mean(homes) >= 0.7),
                   "G-A3(query方位45°>=0.6)": bool(qs and np.mean(qs) >= 0.6)})
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
