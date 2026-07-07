#!/usr/bin/env python3
"""Y2 地形判断探针:冻结 YOLOE 嵌入上,floor/wall/hole 三类地形头可学吗?

命题(用户 2026-07-08):YOLOE 能力能否支撑对地形的正确判断?
考点设计:floor 与 wall 同材质(石头)——区分只能靠几何/明暗/透视,纹理帮不上;
hole=负空间(挖掉地板的沟壕开口),对纹理型嵌入是真考验。
GT 免人工:沟壕挖在已知坐标 → 顶面投影(开口四边形);floor=余下地板顶面;
wall=前墙前脸。判据预登记:留出局 hole mIoU ≥0.5(过了才谈行为级绕沟导航)。

用法:
  --mode collect --episodes 24 ; --mode run(训练+留出评测)
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
import torch

from net.fovea_twotower.seg_head import EYE_H, FOV_V, cam_basis
from net.fovea_twotower.token_stream import as_hwc
from net.fovea_twotower.yolo_unified import PAD_TOP, UnifiedYoloe26, pad384

TCLASSES = ["floor", "wall", "hole"]
W, H = 640, 360


def build_trench_course(trench_z, wall_z=8):
    """石地板房间 + 前墙;沟壕(2 格宽 x∈[-3,3],深 2)挖在 z=trench_z。"""
    return [
        "gamemode survival @p",
        "difficulty peaceful",
        "tp @p ~ ~ ~ 0 0",
        f"fill ~-4 ~-1 ~-2 ~4 ~4 ~{wall_z} minecraft:air",
        f"fill ~-4 ~-2 ~-2 ~4 ~-2 ~{wall_z} minecraft:stone",
        f"fill ~-4 ~-1 ~{wall_z} ~4 ~4 ~{wall_z} minecraft:stone",
        f"fill ~-3 ~-2 ~{trench_z} ~3 ~-4 ~{trench_z + 1} minecraft:air",  # 沟壕
        "clear @p",
    ]


def project_quad(corners, pose):
    """4 角点世界坐标 → pad384 像素点集;任一角在背后 → None。"""
    x, y, z, yaw, pitch = pose
    eye = np.array([x, y + EYE_H, z])
    f, r, u = cam_basis(yaw, pitch)
    fy = (H / 2) / np.tan(np.radians(FOV_V) / 2)
    pts = []
    for c in corners:
        v = np.asarray(c, float) - eye
        zc = v @ f
        if zc < 0.15:
            return None
        pts.append([W / 2 + fy * (v @ r) / zc,
                    H / 2 - fy * (v @ u) / zc + PAD_TOP])
    return np.array(pts, np.float32)


def top_quad(bx, by, bz):
    return [(bx, by + 1, bz), (bx + 1, by + 1, bz),
            (bx + 1, by + 1, bz + 1), (bx, by + 1, bz + 1)]


def front_quad(bx, by, bz):
    return [(bx, by, bz), (bx + 1, by, bz), (bx + 1, by + 1, bz), (bx, by + 1, bz)]


def terrain_masks(gt, pose):
    """gt={cls:[(kind,x,y,z),...]} → {cls: bool[384,640]}。kind: top|front。"""
    out = {}
    for cls, blocks in gt.items():
        m = np.zeros((384, 640), np.uint8)
        for kind, bx, by, bz in blocks:
            q = top_quad(bx, by, bz) if kind == "top" else front_quad(bx, by, bz)
            pts = project_quad(q, pose)
            if pts is None:
                continue
            cv2.fillConvexPoly(m, cv2.convexHull(pts.astype(np.int32)), 1)
        out[cls] = m.astype(bool)
    # hole 压过 floor(开口处不可能同时是地板)
    out["floor"] &= ~out["hole"]
    return out


def run_collect(args):
    from craftground import make
    from craftground.initial_environment_config import (InitialEnvironmentConfig,
                                                        WorldType)
    from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
    from craftground.screen_encoding_modes import ScreenEncodingMode
    from tests.integration.collect_calib640 import ObservePolicy, _pose, _ray

    os.makedirs(args.out, exist_ok=True)
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
    rng = np.random.default_rng(args.seed)
    wall_z = 8
    n = 0
    for ep in range(args.episodes):
        outp = os.path.join(args.out, f"tp_s{args.seed}_e{ep}.npz")
        if os.path.exists(outp):
            continue
        tz = int(rng.integers(2, 6))                    # 沟壕 z 随机
        obs, _ = env.reset(options={"fast_reset": True,
                                    "extra_commands": build_trench_course(tz, wall_z)})
        for _ in range(8):
            obs, *_ = env.step(noop)
        # 锚定:准星正对前墙中央 → raycast 得前墙块绝对坐标,反推整个房间几何
        xyzk = None
        for _ in range(20):
            obs, *_ = env.step(noop)
            xyz, key, _d = _ray(obs["full"])
            if "stone" in key:
                xyzk = xyz
                break
        if xyzk is None:
            print(f"[tp] ✗ e{ep} 锚定失败", flush=True)
            continue
        wx, wy, wz = xyzk                               # 前墙眼平中央块(=feet+1)
        floor_y = wy - 2                                # 地板块 y=feet-1=wy-2
                                                        # (v1 用 wy-1 高一格→整层GT向地平线漂移,洞口画到墙脚)
        gt = {"wall": [("front", wx + dx, wy + dy, wz)
                       for dx in range(-4, 5) for dy in range(-1, 3)],
              "floor": [], "hole": []}
        for dx in range(-4, 5):
            for z in range(wz - wall_z - 1, wz):        # 房间地板 z 范围(相对反推)
                zz = z
                rel_z = zz - (wz - wall_z)              # 房内相对 z(0=玩家附近)
                if -3 <= dx <= 3 and tz <= rel_z <= tz + 1:
                    gt["hole"].append(("top", wx + dx, floor_y, zz))
                else:
                    gt["floor"].append(("top", wx + dx, floor_y, zz))
        pol = ObservePolicy(rng)
        frames, poses = [], []
        for t in range(args.steps):
            a = pol(t, noop, obs)
            obs, *_ = env.step(a)
            if t % 3 == 0:
                frames.append(np.ascontiguousarray(
                    as_hwc(obs["rgb"]).transpose(2, 0, 1)))
                poses.append(_pose(obs["full"]))
        np.savez_compressed(outp, frames=np.stack(frames).astype(np.uint8),
                            pose=np.array(poses, np.float32),
                            gt=json.dumps(gt), meta=json.dumps({"trench_z": tz}))
        n += 1
        print(f"[tp] ✓ e{ep} tz={tz} T={len(frames)}", flush=True)
    env.close()
    print(f"[tp] DONE {n} → {args.out}", flush=True)


def run_train_eval(args):
    from net.fovea_twotower.seg_head import ConvSegHead

    dev = "cuda"
    u = UnifiedYoloe26(device=dev)
    files = sorted(glob.glob(os.path.join(args.data, "*.npz")))
    hold = max(3, len(files) // 5)
    tr_f, te_f = files[:-hold], files[-hold:]

    def frames_labels(fs):
        F_, L_ = [], []
        for f in fs:
            z = np.load(f, allow_pickle=True)
            gt = {k: [tuple(b) for b in v] for k, v in json.loads(str(z["gt"])).items()}
            for i in range(len(z["frames"])):
                img = pad384(z["frames"][i].transpose(1, 2, 0))
                ms = terrain_masks(gt, z["pose"][i])
                lab = np.full((384, 640), len(TCLASSES), np.int64)
                for k, c in enumerate(TCLASSES):
                    lab[ms[c]] = k
                if (lab != len(TCLASSES)).sum() < 500:
                    continue
                F_.append(u.embed(img)[0][0].half().cpu())
                L_.append(torch.from_numpy(lab))
        return F_, L_

    Ftr, Ltr = frames_labels(tr_f)
    print(f"[tp] train {len(Ftr)} 帧", flush=True)
    head = ConvSegHead(ncls=len(TCLASSES) + 1).to(dev)
    cnt = torch.stack([(torch.stack(Ltr) == k).sum()
                       for k in range(len(TCLASSES) + 1)])
    wgt = (cnt.sum() / (cnt.float() + 1)).sqrt()
    wgt = (wgt / wgt.mean()).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    for e in range(args.epochs):
        idx = rng.permutation(len(Ftr))
        tot = 0.0
        for i0 in range(0, len(idx), 8):
            bi = idx[i0:i0 + 8]
            x = torch.stack([Ftr[i] for i in bi]).float().to(dev)
            y = torch.stack([Ltr[i] for i in bi]).to(dev)
            loss = torch.nn.functional.cross_entropy(head(x), y, weight=wgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(bi)
        print(f"[tp] epoch {e} loss={tot/len(idx):.4f}", flush=True)
    torch.save(head.state_dict(), args.head_out)

    # 留出评测
    ious = {c: [] for c in TCLASSES}
    for f in te_f:
        z = np.load(f, allow_pickle=True)
        gt = {k: [tuple(b) for b in v] for k, v in json.loads(str(z["gt"])).items()}
        for i in range(0, len(z["frames"]), 2):
            img = pad384(z["frames"][i].transpose(1, 2, 0))
            ms = terrain_masks(gt, z["pose"][i])
            with torch.no_grad():
                lab = head(u.embed(img)[0].float())[0].argmax(0).cpu().numpy()
            for k, c in enumerate(TCLASSES):
                if ms[c].sum() >= 400:
                    pred = lab == k
                    inter = (pred & ms[c]).sum()
                    union = (pred | ms[c]).sum()
                    if union:
                        ious[c].append(inter / union)
    res = {c: (round(float(np.mean(v)), 3) if v else None) for c, v in ious.items()}
    verdict = "PASS" if (res.get("hole") or 0) >= 0.5 else "FAIL"
    out = dict(miou=res, gate="hole>=0.5", verdict=verdict,
               n_train_frames=len(Ftr))
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[tp] {json.dumps(out, ensure_ascii=False)}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["collect", "run"], required=True)
    p.add_argument("--data", default="runs/data/terrain")
    p.add_argument("--out", default="runs/data/terrain")
    p.add_argument("--episodes", type=int, default=24)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--head_out", default="runs/terrain_head.pt")
    p.add_argument("--out_json", default="runs/terrain_probe.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--port", type=int, default=8690)
    args = p.parse_args()
    (run_collect if args.mode == "collect" else run_train_eval)(args)


if __name__ == "__main__":
    main()
