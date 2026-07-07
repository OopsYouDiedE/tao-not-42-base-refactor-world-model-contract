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

C = len(CLASSES)
# 真实感知噪声下峰接受更严(1.15→1.3):错误修正比漏修正贵(smoke 教训)
# 修正增益按卡尔曼直觉压低:真实感知定位噪声(~3-5格)>>短程死算漂移(~1格),
# 单次修正不可信,低阻尼×高频=噪声平均、信号积累
RELOC_CFG = dict(window=6, min_ratio=1.3, subcell=False, min_pts=3.0)
DAMP = 0.35


def yaw_dir(yaw_deg):
    """MC 朝向单位向量(x,z):yaw0=+z(南),+yaw 向 -x。"""
    r = np.radians(yaw_deg)
    return np.array([-np.sin(r), np.cos(r)])


def det_world_xz(tok, pose, cal):
    """token → 世界系相对 (x,z):方位=yaw+拟合斜率·像素偏移,距离=k/√面积。

    斜率必须标定拟合:0.15°/px 是相机动作口径非屏幕几何(smoke 教训,
    残差 19.3°);距离>write_maxd 不写图(径向误差随距放大,超相关窗窗宽)。"""
    cx_px, area = tok[0] * 640, max(tok[4] * 640 * 384, 1.0)
    brg = pose[3] + cal["deg_per_px"] * (cx_px - 320)
    d = cal["area_k"] / np.sqrt(area)
    if d > cal.get("write_maxd", 12.0):
        return None
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
            # 配对按方位消歧:候选=该类全部 GT 块;粗斜率 0.15 选最近方位,
            # 次近方位差 <10° 的歧义帧丢弃(smoke 教训:按距离配对错配污染拟合)
            cx_px = tok[0] * 640
            rough = pose[3] + 0.15 * (cx_px - 320)
            cands = []
            for b in gt[CLASSES[ci]]:
                rel = np.array([b[0] + .5 - pose[0], b[2] + .5 - pose[2]])
                gt_brg = np.degrees(np.arctan2(-rel[0], rel[1]))    # MC yaw 系
                dbrg = abs((gt_brg - rough + 180) % 360 - 180)
                cands.append((dbrg, rel, gt_brg))
            if not cands:
                continue
            cands.sort(key=lambda c: c[0])
            if cands[0][0] > 30 or (len(cands) > 1 and cands[1][0] - cands[0][0] < 10):
                continue
            _, rel, gt_brg = cands[0]
            gt_d = float(np.linalg.norm(rel))
            rel_brg = (gt_brg - pose[3] + 180) % 360 - 180
            if gt_d < 25:
                pairs.append((tok[4] * 640 * 384, gt_d, cx_px, rel_brg))
        obs, *_ = env.step(a)
    sp = np.sort([x for x in speeds if x > 0.02])
    v = float(np.median(sp[int(0.4 * len(sp)):]))   # 上60%中位:撞墙滑行步会
                                                    # 拖低全体中位(seed23 教训 0.11/0.16)
    ar = np.array(pairs)
    k = float(np.median(ar[:, 1] * np.sqrt(ar[:, 0])))
    px = ar[:, 2] - 320
    slope = float(np.sum(px * ar[:, 3]) / np.sum(px * px))      # 过原点最小二乘
    resid = float(np.median(np.abs(ar[:, 3] - slope * px)))
    dist_resid = float(np.median(np.abs(k / np.sqrt(ar[:, 0]) - ar[:, 1])))
    return dict(v=v, area_k=k, deg_per_px=slope, n_pairs=len(pairs),
                brg_resid=resid, dist_resid=dist_resid, write_maxd=10.0)


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
    # 双域:干净臂(m_clean,纯死算,验集成交付=回家/查询);退化臂(m_noisy,
    # 里程计丢步p=0.15+增益噪声0.2=MC粗糙地形/频繁碰撞域,验自定位机制)
    m_clean = EgoMapNorthLoc(C, 128, 32.0)
    m_noisy = EgoMapNorthLoc(C, 128, 32.0)
    p_dead = np.zeros(2)
    pn_dead, pn_reloc = np.zeros(2), np.zeros(2)
    e_dead, en_dead, en_reloc = [], [], []
    prev_fwd_yaw, slip = None, False
    for t in range(steps):
        goal = int(g1 if t < switch_t else g2)
        pose = _pose(obs["full"])
        p_gt = np.array([pose[0], pose[2]]) - origin
        # —— 里程计:上一步按下 forward 则记 v̂·dir(yaw);碰撞漂移天然存在 ——
        if prev_fwd_yaw is not None:
            d_odo = cal["v"] * yaw_dir(prev_fwd_yaw)
            p_dead += d_odo
            m_clean.step(d_odo)
            # 突发滑移:进入p=0.04/退出p=0.3,滑移期里程计只记30%位移
            # (零均值独立丢步被往返轨迹抵消,seed23 教训 ndead≈dead)
            if slip:
                slip = rng.random() > 0.3
            else:
                slip = rng.random() < 0.08
            dn = d_odo * (0.1 if slip else 1.0) * (1 + rng.normal(0, 0.3, 2))
            pn_dead += dn
            pn_reloc += dn
            m_noisy.step(dn)
        toks = tok_head(as_hwc(obs["rgb"]))
        pts, fts = [], []
        for tok in toks:
            ci = int(np.argmax(tok[6:6 + C]))
            if tok[6 + ci] < 0.4 or tok[4] <= 0:
                continue
            p = det_world_xz(tok, pose, cal)
            if p is None:
                continue
            pts.append(p)
            fts.append(np.eye(C)[ci])
        if pts:
            pts_t = torch.from_numpy(np.array(pts)).float()
            fts_t = torch.from_numpy(np.array(fts)).float()
            if t % every == 0 and t > 10:
                e_hat = m_noisy.relocalize(pts_t, fts_t, **RELOC_CFG)
                if e_hat is not None:
                    pn_reloc += DAMP * e_hat
                    m_noisy.step(DAMP * e_hat)
            m_clean.write(pts_t, fts_t)
            m_noisy.write(pts_t, fts_t)
        e_dead.append(float(np.linalg.norm(p_dead - p_gt)))
        en_dead.append(float(np.linalg.norm(pn_dead - p_gt)))
        en_reloc.append(float(np.linalg.norm(pn_reloc - p_gt)))
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
    home_ok = brg_ok(-p_dead, -p_gt)          # 集成交付=干净臂纯死算
    rels = [np.array([b[0] + .5 - pose[0], b[2] + .5 - pose[2]])
            for b in gt[CLASSES[int(g2)]]]
    q_ok = q_any = None
    if rels:
        gt_rel = min(rels, key=np.linalg.norm)
        v, txt = MapQuery(m_clean, CLASSES).nearest(CLASSES[int(g2)])
        if v is not None:
            vv = np.array([float(v[0]), float(v[1])])   # 已是相对自身(图中心)向量
                                                        # smoke4 教训:勿再减 p̂
            q_ok = brg_ok(vv, gt_rel)                   # 严:指向 GT 最近实例
            q_any = any(brg_ok(vv, r) for r in rels)    # 宽:指向任一真实实例
                                                        # (区分"最近实例歧义"vs"图错")
    return dict(dead_end=e_dead[-1],
                ndead_end=en_dead[-1], nreloc_end=en_reloc[-1],
                home_ok=home_ok, query_ok=q_ok, query_any=q_any)


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
            print(f"[a1] ep{ep} dead={r['dead_end']:.2f} "
                  f"n_dead={r['ndead_end']:.2f} n_reloc={r['nreloc_end']:.2f} "
                  f"home={'✓' if r['home_ok'] else '✗' if r['home_ok'] is not None else '-'} "
                  f"query={'✓' if r['query_ok'] else '✗' if r['query_ok'] is not None else '-'}",
                  flush=True)
    env.close()
    dead = float(np.median([r["dead_end"] for r in ms]))
    ndead = float(np.median([r["ndead_end"] for r in ms]))
    nreloc = float(np.median([r["nreloc_end"] for r in ms]))
    homes = [r["home_ok"] for r in ms if r["home_ok"] is not None]
    qs = [r["query_ok"] for r in ms if r["query_ok"] is not None]
    qa = [r["query_any"] for r in ms if r["query_any"] is not None]
    out = dict(n=len(ms), dead_end_med=round(dead, 2),
               noisy_dead_end_med=round(ndead, 2),
               noisy_reloc_end_med=round(nreloc, 2),
               home_rate=round(float(np.mean(homes)), 2) if homes else None,
               query_rate=round(float(np.mean(qs)), 2) if qs else None,
               query_any_rate=round(float(np.mean(qa)), 2) if qa else None,
               calib=cal,
               gates={
                   "G-A1(集成:home>=0.7且query>=0.6)":
                       bool(homes and qs and np.mean(homes) >= 0.7
                            and np.mean(qs) >= 0.6),
                   "G-A2(自定位高漂移:nreloc<=0.6×ndead)":
                       bool(nreloc <= 0.6 * ndead)})
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
