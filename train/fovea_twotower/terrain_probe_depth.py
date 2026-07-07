#!/usr/bin/env python3
"""Y2c 地形闭环:深度通道充分性 oracle 上界——加解析深度后负空间可学吗?

Y2(YOLOE 0.266)/Y2b(DINO 0.353)已证 RGB 单帧嵌入是负空间盲区的通病。
本探针注入特权几何:房间占用格已知(GT 工厂同源),逐帧解析光线步进出 P3 网格
深度图(引擎深度缓冲在 llvmpipe 软件渲染下退化恒 1.0,不可用,实测 07-08)。

四臂(先登记):
  embdepth: YOLOE emb(512) ⊕ log 深度(1) → ConvSegHead(cin=513)
  depthonly: 仅深度(cin=1)——洞若纯几何可判,此臂应独立过门
  heightonly/embheight: 深度换反投影高度 h=dir_y·t(命中点相对眼睛高度)
判据同 Y2:留出 hole mIoU ≥0.5。
深度臂结果(07-08):双 FAIL(0.156/0.31)——原始深度非判别特征:conv 头平移不变
无位置先验,"深"=洞 or 远地板不可分;掠射角下洞的深度不连续性趋零。
高度臂动机:h 平移不变(地板恒 -1.62/洞更低/墙更高),且可部署(单目深度+内参
反投影,无特权)。PASS ⇒ 几何通道的正确形式=高度图,路径=深度蒸馏+反投影。
注意:解析几何为特权输入(训练+评测同享),本探针测"充分性/形式"非"可部署性"。
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from net.fovea_twotower.seg_head import EYE_H, FOV_V, cam_basis, ConvSegHead
from net.fovea_twotower.yolo_unified import PAD_TOP, UnifiedYoloe26, pad384
from train.fovea_twotower.terrain_probe import TCLASSES, terrain_masks

W, H = 640, 360
GW, GH = 80, 48                 # P3 网格(stride 8 @ 640×384)
TMAX, TSTEP = 40.0, 0.08


def analytic_depth(pose, wz, floor_y, hole_xz):
    """已知占用格 + 位姿 → P3 网格 [48,80] 光线首次命中距离(log1p 归一)。

    占用:z≥wz 前墙面;洞列(x,z)∈hole_xz 在 y≤floor_y-3 才实;余下 y≤floor_y 实。"""
    x0, y0, z0, yaw, pitch = pose
    eye = np.array([x0, y0 + EYE_H, z0])
    f, r, u = cam_basis(yaw, pitch)
    fy = (H / 2) / np.tan(np.radians(FOV_V) / 2)
    # P3 cell 中心(pad384 坐标) → 相机光线
    px = (np.arange(GW) + 0.5) * 8.0
    py = (np.arange(GH) + 0.5) * 8.0
    PX, PY = np.meshgrid(px, py)
    dirs = (f[None, None]
            + r[None, None] * ((PX - W / 2) / fy)[..., None]
            + u[None, None] * ((H / 2 - (PY - PAD_TOP)) / fy)[..., None])
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    depth = np.full((GH, GW), TMAX, np.float32)
    hrel = np.full((GH, GW), 0.0, np.float32)      # 命中点相对眼睛高度(反投影,可部署)
    alive = np.ones((GH, GW), bool)
    hkeys = np.array([bx * 100000 + bz for bx, bz in hole_xz], np.int64)
    for t in np.arange(0.3, TMAX, TSTEP):
        if not alive.any():
            break
        p = eye[None, None] + dirs * t
        bx = np.floor(p[..., 0]).astype(np.int64)
        by = np.floor(p[..., 1]).astype(np.int64)
        bz = np.floor(p[..., 2]).astype(np.int64)
        wall = bz >= wz
        ground = by <= floor_y
        if hkeys.size:
            in_hole = np.isin(bx * 100000 + bz, hkeys)
            ground = np.where(in_hole, by <= floor_y - 3, ground)
        hit = alive & (wall | ground)
        depth[hit] = t
        hrel[hit] = (dirs[..., 1] * t)[hit]
        alive &= ~hit
    return np.log1p(depth) / np.log1p(TMAX), np.clip(hrel / 4.0, -1, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="runs/data/terrain_v2")
    p.add_argument("--arm", choices=["embdepth", "depthonly",
                                     "heightonly", "embheight"], required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--out_json", default="")
    args = p.parse_args()
    out_json = args.out_json or f"runs/terrain_probe_{args.arm}.json"

    dev = "cuda"
    u = (UnifiedYoloe26(device=dev)
         if args.arm in ("embdepth", "embheight") else None)
    files = sorted(glob.glob(os.path.join(args.data, "*.npz")))
    hold = max(3, len(files) // 5)
    tr_f, te_f = files[:-hold], files[-hold:]   # 与 Y2/Y2b 同一切分

    def feat(img_pad, dmap, hmap):
        g = dmap if "depth" in args.arm else hmap
        d = torch.from_numpy(g)[None].float()
        if args.arm in ("depthonly", "heightonly"):
            return d
        e = u.embed(img_pad)[0][0].float().cpu()
        return torch.cat([e, d], 0)

    def load(fs):
        F_, L_ = [], []
        for fp in fs:
            z = np.load(fp, allow_pickle=True)
            gt = {k: [tuple(b) for b in v]
                  for k, v in json.loads(str(z["gt"])).items()}
            wz = gt["wall"][0][3]
            floor_y = gt["floor"][0][2]
            hole_xz = {(b[1], b[3]) for b in gt["hole"]}
            for i in range(len(z["frames"])):
                img = pad384(z["frames"][i].transpose(1, 2, 0))
                ms = terrain_masks(gt, z["pose"][i])
                lab = np.full((384, 640), len(TCLASSES), np.int64)
                for k, c in enumerate(TCLASSES):
                    lab[ms[c]] = k
                if (lab != len(TCLASSES)).sum() < 500:
                    continue
                dm, hm = analytic_depth(z["pose"][i], wz, floor_y, hole_xz)
                F_.append(feat(img, dm, hm).half())
                L_.append(torch.from_numpy(lab))
        return F_, L_

    Ftr, Ltr = load(tr_f)
    cin = Ftr[0].shape[0]
    print(f"[tpc:{args.arm}] train {len(Ftr)} 帧 cin={cin}", flush=True)
    head = ConvSegHead(cin=cin, ncls=len(TCLASSES) + 1).to(dev)
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
            loss = F.cross_entropy(head(x), y, weight=wgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(bi)
        print(f"[tpc:{args.arm}] epoch {e} loss={tot/len(idx):.4f}", flush=True)

    Fte, Lte = load(te_f)
    ious = {c: [] for c in TCLASSES}
    with torch.no_grad():
        for x, y in zip(Fte, Lte):
            lab = head(x[None].float().to(dev))[0].argmax(0).cpu().numpy()
            yn = y.numpy()
            for k, c in enumerate(TCLASSES):
                m = yn == k
                if m.sum() >= 400:
                    pred = lab == k
                    union = (pred | m).sum()
                    if union:
                        ious[c].append((pred & m).sum() / union)
    res = {c: (round(float(np.mean(v)), 3) if v else None)
           for c, v in ious.items()}
    verdict = "PASS" if (res.get("hole") or 0) >= 0.5 else "FAIL"
    out = dict(arm=args.arm, miou=res, gate="hole>=0.5", verdict=verdict,
               privileged="analytic depth (oracle上界,非部署形态)",
               n_train_frames=len(Ftr))
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[tpc:{args.arm}] {json.dumps(out, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
