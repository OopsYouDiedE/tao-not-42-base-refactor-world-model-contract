#!/usr/bin/env python3
"""Y2d 地形部署收口:RGB+自身俯仰 → 高度图蒸馏,预测高度重跑 heightonly 协议。

Y2c 已证反投影高度是负空间的正确几何表征(oracle 0.807 PASS),但 oracle 高度
来自解析几何=特权。本实验测部署形态:HeightNet(小 CNN)从 RGB(384×640)+俯仰角
(本体感受通道,agent 自身动作状态,非特权)预测 P3 网格高度图,解析高度只做
训练监督(=深度蒸馏的高度形式);下游 ConvSegHead 训练与评测全部吃预测高度。

判据(先登记,与 Y2 系同门):留出局 hole mIoU ≥0.5。
PASS ⇒ 地形判断可部署收口:快塔地形支路 = HeightNet→高度图→地形头,零特权;
FAIL ⇒ 单帧单目高度蒸馏不足,升级多帧视差/更强深度先验(蒸馏现成单目深度模型)。
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from net.fovea_twotower.seg_head import ConvSegHead
from net.fovea_twotower.yolo_unified import pad384
from train.fovea_twotower.terrain_probe import TCLASSES, terrain_masks
from train.fovea_twotower.terrain_probe_depth import analytic_depth


class HeightNet(nn.Module):
    """RGB[3,384,640]+pitch 广播通道 → 高度图 [1,48,80](tanh,对齐 clip±1)。~0.5M。"""

    def __init__(self, ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, ch, 5, 2, 2), nn.GELU(),          # /2
            nn.Conv2d(ch, ch * 2, 3, 2, 1), nn.GELU(),     # /4
            nn.Conv2d(ch * 2, ch * 4, 3, 2, 1), nn.GELU(), # /8 → 48×80
            nn.Conv2d(ch * 4, ch * 4, 3, 1, 1), nn.GELU(),
            nn.Conv2d(ch * 4, 1, 1))

    def forward(self, rgb, pitch_deg):                     # rgb [B,3,384,640]
        pc = (pitch_deg / 90.0).view(-1, 1, 1, 1).expand(-1, 1, *rgb.shape[2:])
        return torch.tanh(self.net(torch.cat([rgb, pc], 1)))


def load_split(data):
    files = sorted(glob.glob(os.path.join(data, "*.npz")))
    hold = max(3, len(files) // 5)
    return files[:-hold], files[-hold:]                    # 与 Y2 系同一切分


def load_frames(fs):
    """→ list of (rgb_u8[3,384,640], pitch, hmap[48,80], lab[384,640])。"""
    out = []
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
            _, hm = analytic_depth(z["pose"][i], wz, floor_y, hole_xz)
            out.append((img.transpose(2, 0, 1).copy(), float(z["pose"][i][4]),
                        hm, lab))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="runs/data/terrain_v2")
    p.add_argument("--h_epochs", type=int, default=20)
    p.add_argument("--s_epochs", type=int, default=10)
    p.add_argument("--out_json", default="runs/terrain_height_distill.json")
    p.add_argument("--net_out", default="runs/terrain_heightnet.pt")
    args = p.parse_args()

    dev = "cuda"
    tr_f, te_f = load_split(args.data)
    TR, TE = load_frames(tr_f), load_frames(te_f)
    print(f"[y2d] train {len(TR)} / holdout {len(TE)} 帧", flush=True)

    # —— 阶段1:高度蒸馏 ——
    hnet = HeightNet().to(dev)
    opt = torch.optim.AdamW(hnet.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    for e in range(args.h_epochs):
        idx = rng.permutation(len(TR))
        tot = 0.0
        for i0 in range(0, len(idx), 8):
            bi = idx[i0:i0 + 8]
            rgb = torch.stack([torch.from_numpy(TR[i][0]) for i in bi]
                              ).float().to(dev) / 255
            pit = torch.tensor([TR[i][1] for i in bi], device=dev)
            tgt = torch.stack([torch.from_numpy(TR[i][2]) for i in bi]
                              )[:, None].float().to(dev)
            loss = F.huber_loss(hnet(rgb, pit), tgt, delta=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(bi)
        print(f"[y2d] hnet epoch {e} loss={tot/len(idx):.5f}", flush=True)
    torch.save(hnet.state_dict(), args.net_out)
    hnet.eval()

    @torch.no_grad()
    def pred_h(sample):
        rgb = torch.from_numpy(sample[0])[None].float().to(dev) / 255
        pit = torch.tensor([sample[1]], device=dev)
        return hnet(rgb, pit)[0].cpu()                     # [1,48,80]

    # 高度回归质量(留出,诊断用非门)
    errs = [float((pred_h(s)[0] - torch.from_numpy(s[2])).abs().mean())
            for s in TE]
    h_mae = float(np.mean(errs))
    print(f"[y2d] holdout 高度 MAE={h_mae:.4f}(clip±1 尺度)", flush=True)

    # —— 阶段2:地形头训练+评测全吃预测高度 ——
    Ftr = [pred_h(s).half() for s in TR]
    Ltr = [torch.from_numpy(s[3]) for s in TR]
    head = ConvSegHead(cin=1, ncls=len(TCLASSES) + 1).to(dev)
    cnt = torch.stack([(torch.stack(Ltr) == k).sum()
                       for k in range(len(TCLASSES) + 1)])
    wgt = (cnt.sum() / (cnt.float() + 1)).sqrt()
    wgt = (wgt / wgt.mean()).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)
    for e in range(args.s_epochs):
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
        print(f"[y2d] seg epoch {e} loss={tot/len(idx):.4f}", flush=True)

    ious = {c: [] for c in TCLASSES}
    with torch.no_grad():
        for s in TE:
            lab = head(pred_h(s)[None].float().to(dev))[0].argmax(0).cpu().numpy()
            for k, c in enumerate(TCLASSES):
                m = s[3] == k
                if m.sum() >= 400:
                    pred = lab == k
                    union = (pred | m).sum()
                    if union:
                        ious[c].append((pred & m).sum() / union)
    res = {c: (round(float(np.mean(v)), 3) if v else None)
           for c, v in ious.items()}
    verdict = "PASS" if (res.get("hole") or 0) >= 0.5 else "FAIL"
    out = dict(miou=res, gate="hole>=0.5", verdict=verdict,
               height_mae=round(h_mae, 4), n_train_frames=len(TR),
               inputs="RGB+pitch(本体感受),无特权")
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[y2d] {json.dumps(out, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
