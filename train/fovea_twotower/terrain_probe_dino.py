#!/usr/bin/env python3
"""Y2b 地形归因分岔:同协议换 DINOv2 patch 特征——负空间盲区是 YOLOE 特有还是 RGB 嵌入通病?

Y2(terrain_probe.py)裁决:冻结 YOLOE P3 上 hole mIoU 0.266 FAIL(门 0.5)。
本探针唯一变量=特征来源:DINOv2-S/14 patch 网格(644×392 输入→28×46 格,自监督
通用特征,含明暗/几何线索的可能性高于检测式校准嵌入)。数据复用 terrain_v2,零采集。

裁决分岔(先登记):
  DINO hole ≥0.5 → 盲区是 YOLOE 特有,快塔地形走 DINO 支路(慢塔已有同款骨干,零新件);
  DINO 也 <0.5 → RGB 单帧嵌入通病坐实,深度/几何通道升为必需。
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from net.fovea_twotower.seg_head import ConvSegHead
from net.fovea_twotower.yolo_unified import pad384
from train.fovea_twotower.terrain_probe import TCLASSES, terrain_masks

GW, GH = 46, 28                         # 644//14, 392//14
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DinoGrid:
    def __init__(self, dev):
        from net.backbone import build_backbone
        from net.config import BackboneConfig
        self.m, _, self.dim, _, _ = build_backbone(BackboneConfig(kind="dinov2"))
        self.m = self.m.to(dev).eval()
        self.dev = dev

    @torch.no_grad()
    def __call__(self, img_pad384_hwc_u8):
        x = torch.from_numpy(np.ascontiguousarray(img_pad384_hwc_u8))
        x = x.permute(2, 0, 1)[None].float().to(self.dev) / 255
        x = F.interpolate(x, size=(GH * 14, GW * 14), mode="bilinear",
                          align_corners=False)
        x = (x - MEAN.to(self.dev)) / STD.to(self.dev)
        t = self.m(pixel_values=x).last_hidden_state[:, 1:]     # [1,GH*GW,384]
        return t.permute(0, 2, 1).reshape(1, self.dim, GH, GW)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="runs/data/terrain_v2")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--head_out", default="runs/terrain_head_dino.pt")
    p.add_argument("--out_json", default="runs/terrain_probe_dino.json")
    args = p.parse_args()

    dev = "cuda"
    enc = DinoGrid(dev)
    files = sorted(glob.glob(os.path.join(args.data, "*.npz")))
    hold = max(3, len(files) // 5)
    tr_f, te_f = files[:-hold], files[-hold:]   # 与 Y2 同一留出切分

    def frames_labels(fs):
        F_, L_ = [], []
        for f in fs:
            z = np.load(f, allow_pickle=True)
            gt = {k: [tuple(b) for b in v]
                  for k, v in json.loads(str(z["gt"])).items()}
            for i in range(len(z["frames"])):
                img = pad384(z["frames"][i].transpose(1, 2, 0))
                ms = terrain_masks(gt, z["pose"][i])
                lab = np.full((384, 640), len(TCLASSES), np.int64)
                for k, c in enumerate(TCLASSES):
                    lab[ms[c]] = k
                if (lab != len(TCLASSES)).sum() < 500:
                    continue
                F_.append(enc(img)[0].half().cpu())
                L_.append(torch.from_numpy(lab))
        return F_, L_

    Ftr, Ltr = frames_labels(tr_f)
    print(f"[tpd] train {len(Ftr)} 帧 dim={enc.dim} grid={GH}x{GW}", flush=True)
    head = ConvSegHead(cin=enc.dim, ncls=len(TCLASSES) + 1).to(dev)
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
        print(f"[tpd] epoch {e} loss={tot/len(idx):.4f}", flush=True)
    torch.save(head.state_dict(), args.head_out)

    ious = {c: [] for c in TCLASSES}
    near, far = [], []
    for f in te_f:
        z = np.load(f, allow_pickle=True)
        gt = {k: [tuple(b) for b in v]
              for k, v in json.loads(str(z["gt"])).items()}
        for i in range(0, len(z["frames"]), 2):
            img = pad384(z["frames"][i].transpose(1, 2, 0))
            ms = terrain_masks(gt, z["pose"][i])
            with torch.no_grad():
                lab = head(enc(img).float())[0].argmax(0).cpu().numpy()
            for k, c in enumerate(TCLASSES):
                if ms[c].sum() >= 400:
                    pred = lab == k
                    union = (pred | ms[c]).sum()
                    if union:
                        iou = (pred & ms[c]).sum() / union
                        ious[c].append(iou)
                        if c == "hole":
                            cy = np.argwhere(ms[c])[:, 0].mean()
                            (near if cy > 250 else far).append(iou)
    res = {c: (round(float(np.mean(v)), 3) if v else None)
           for c, v in ious.items()}
    verdict = "PASS" if (res.get("hole") or 0) >= 0.5 else "FAIL"
    out = dict(miou=res, gate="hole>=0.5", verdict=verdict, encoder="dinov2-s14",
               hole_near=round(float(np.mean(near)), 3) if near else None,
               hole_far=round(float(np.mean(far)), 3) if far else None,
               n_train_frames=len(Ftr))
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[tpd] {json.dumps(out, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
