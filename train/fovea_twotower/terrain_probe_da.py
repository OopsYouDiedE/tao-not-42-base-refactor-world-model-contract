#!/usr/bin/env python3
"""Y2e 地形部署收口二试:预训练单目深度先验(Depth Anything V2 metric)→ 反投影高度。

Y2d 已证从零小网学不出遮挡先验(洞盲复发)。本探针换"真实世界深度先验零样本
迁移":DA-V2 metric 预测米制 z-深度 → 已知内参+自身俯仰(本体感受)反投影
命中点相对眼高 h = d·(-sin p + y_n·cos p) → clip(h/4,±1) → heightonly 同协议
(ConvSegHead cin=1,训练评测全吃预测高度)。全链零特权,判据同门 hole ≥0.5。

PASS ⇒ 地形支路定型:DA-small(24M)→高度图→地形头,即插即用;
FAIL ⇒ MC 域差过大,登记路径=用解析高度 GT 微调 DA(域内蒸馏)。
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from net.fovea_twotower.seg_head import ConvSegHead, FOV_V
from train.fovea_twotower.terrain_probe import TCLASSES, terrain_masks
from train.fovea_twotower.terrain_probe_depth import analytic_depth

W, H = 640, 360
PAD_TOP = 12
GW, GH = 80, 48


class DAHeight:
    """RGB[360,640,3] u8 → P3 网格 [48,80] 反投影相对眼高(clip±1)。"""

    def __init__(self, model_id, dev):
        from transformers import (AutoImageProcessor,
                                  AutoModelForDepthEstimation)
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.m = AutoModelForDepthEstimation.from_pretrained(model_id
                                                             ).to(dev).eval()
        self.dev = dev
        f_px = (H / 2) / np.tan(np.radians(FOV_V) / 2)
        px = np.arange(W) + 0.5
        py = np.arange(H) + 0.5
        PX, PY = np.meshgrid(px, py)
        self.y_n = ((H / 2 - PY) / f_px).astype(np.float32)   # 归一像面纵坐标

    @torch.no_grad()
    def __call__(self, rgb_hwc_u8, pitch_deg):
        inp = self.proc(images=rgb_hwc_u8, return_tensors="pt").to(self.dev)
        d = self.m(**inp).predicted_depth                     # [1,h',w'] 米制
        d = F.interpolate(d[None], size=(H, W), mode="bilinear",
                          align_corners=False)[0, 0].cpu().numpy()
        p = np.radians(pitch_deg)                             # MC:正=低头
        h = d * (-np.sin(p) + self.y_n * np.cos(p))
        canvas = np.zeros((384, W), np.float32)
        canvas[PAD_TOP:PAD_TOP + H] = h
        canvas[:PAD_TOP] = h[0]
        canvas[PAD_TOP + H:] = h[-1]
        hp = torch.from_numpy(canvas)[None, None]
        hp = F.avg_pool2d(hp, 8)[0, 0].numpy()                # [48,80]
        return np.clip(hp / 4.0, -1, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="runs/data/terrain_v2")
    p.add_argument("--model", default=
                   "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--out_json", default="runs/terrain_probe_da.json")
    args = p.parse_args()

    dev = "cuda"
    enc = DAHeight(args.model, dev)
    files = sorted(glob.glob(os.path.join(args.data, "*.npz")))
    hold = max(3, len(files) // 5)
    tr_f, te_f = files[:-hold], files[-hold:]   # 与 Y2 系同一切分

    def load(fs, with_gt_h=False):
        F_, L_, err = [], [], []
        for fp in fs:
            z = np.load(fp, allow_pickle=True)
            gt = {k: [tuple(b) for b in v]
                  for k, v in json.loads(str(z["gt"])).items()}
            wz = gt["wall"][0][3]
            floor_y = gt["floor"][0][2]
            hole_xz = {(b[1], b[3]) for b in gt["hole"]}
            for i in range(len(z["frames"])):
                rgb = z["frames"][i].transpose(1, 2, 0)
                ms = terrain_masks(gt, z["pose"][i])
                lab = np.full((384, 640), len(TCLASSES), np.int64)
                for k, c in enumerate(TCLASSES):
                    lab[ms[c]] = k
                if (lab != len(TCLASSES)).sum() < 500:
                    continue
                hp = enc(rgb, float(z["pose"][i][4]))
                if with_gt_h:
                    _, hg = analytic_depth(z["pose"][i], wz, floor_y, hole_xz)
                    err.append(float(np.abs(hp - hg).mean()))
                F_.append(torch.from_numpy(hp)[None].half())
                L_.append(torch.from_numpy(lab))
        return F_, L_, err

    Ftr, Ltr, _ = load(tr_f)
    print(f"[y2e] train {len(Ftr)} 帧", flush=True)
    head = ConvSegHead(cin=1, ncls=len(TCLASSES) + 1).to(dev)
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
        print(f"[y2e] epoch {e} loss={tot/len(idx):.4f}", flush=True)

    Fte, Lte, herr = load(te_f, with_gt_h=True)
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
    out = dict(miou=res, gate="hole>=0.5", verdict=verdict,
               model=args.model,
               height_mae_vs_analytic=round(float(np.mean(herr)), 4),
               n_train_frames=len(Ftr), inputs="RGB+pitch,零特权")
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[y2e] {json.dumps(out, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
