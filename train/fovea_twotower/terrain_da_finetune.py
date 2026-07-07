#!/usr/bin/env python3
"""Y2f 地形部署收口三试:DA-small 域内微调——预训练几何先验 + 解析高度监督。

Y2d(从零 0.5M 网,洞盲)/Y2e(DA 零样本,沟壕凹陷缺失+内参失配)双 FAIL 后的
合取假设:**先验和监督都要**。做法:把 DA-V2-small 的输出通道重解释为
"相对眼高",全参微调(24M,3090 放得下),监督=解析高度(P3 网格,GT 工厂同源),
俯仰角以广播通道注入首层前(拼在 RGB 后过 1×1 适配)。
下游同 heightonly 协议:ConvSegHead 训练评测全吃微调 DA 的预测高度。
判据同门:留出局 hole mIoU ≥0.5。全链部署零特权(RGB+自身俯仰)。

PASS ⇒ 地形支路定型:DA-small(微调)→高度图→地形头;
FAIL ⇒ 登记升级:更多数据(terrain_v3 采集中)/更大 DA/多帧视差。
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
from train.fovea_twotower.terrain_probe import TCLASSES, terrain_masks
from train.fovea_twotower.terrain_probe_depth import analytic_depth

W, H = 640, 360
PAD_TOP = 12
GW, GH = 80, 48
DA_IN = 518                     # DA 标准输入边长(patch14 倍数)


class DAHeightFT(nn.Module):
    """DA-small 全参微调 → 相对眼高 [48,80]。pitch 经 FiLM 加在骨干输入上。"""

    def __init__(self, model_id):
        super().__init__()
        from transformers import AutoModelForDepthEstimation
        self.da = AutoModelForDepthEstimation.from_pretrained(model_id)
        self.pitch_film = nn.Linear(2, 6)      # → RGB 逐通道 scale/shift
        nn.init.zeros_(self.pitch_film.weight)
        nn.init.zeros_(self.pitch_film.bias)
        # 输出适配:预训练米制深度值域 4–75,tanh 会全饱和(v1 教训)→可学仿射
        self.out_scale = nn.Parameter(torch.tensor(0.05))
        self.out_bias = nn.Parameter(torch.tensor(-0.5))
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]
                                              ).view(1, 3, 1, 1), False)
        self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]
                                             ).view(1, 3, 1, 1), False)

    def forward(self, rgb_u8, pitch_deg):      # rgb [B,3,360,640]
        x = rgb_u8.float() / 255
        x = F.interpolate(x, size=(DA_IN, DA_IN), mode="bilinear",
                          align_corners=False)
        x = (x - self.mean) / self.std
        p = torch.stack([torch.sin(torch.deg2rad(pitch_deg)),
                         torch.cos(torch.deg2rad(pitch_deg))], -1)
        f = self.pitch_film(p).view(-1, 6, 1, 1)
        x = x * (1 + f[:, :3]) + f[:, 3:]
        d = self.da(pixel_values=x).predicted_depth[:, None]   # [B,1,h,w]
        d = F.interpolate(d, size=(GH, GW), mode="bilinear",
                          align_corners=False)
        return self.out_scale * d + self.out_bias              # 无饱和,Huber 兜底


def load_frames(fs):
    out = []
    for fp in fs:
        z = np.load(fp, allow_pickle=True)
        gt = {k: [tuple(b) for b in v]
              for k, v in json.loads(str(z["gt"])).items()}
        wz = gt["wall"][0][3]
        floor_y = gt["floor"][0][2]
        hole_xz = {(b[1], b[3]) for b in gt["hole"]}
        for i in range(len(z["frames"])):
            ms = terrain_masks(gt, z["pose"][i])
            lab = np.full((384, 640), len(TCLASSES), np.int64)
            for k, c in enumerate(TCLASSES):
                lab[ms[c]] = k
            if (lab != len(TCLASSES)).sum() < 500:
                continue
            _, hm = analytic_depth(z["pose"][i], wz, floor_y, hole_xz)
            out.append((z["frames"][i].copy(), float(z["pose"][i][4]),
                        hm, lab))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="runs/data/terrain_v2")
    p.add_argument("--extra_data", default="")
    p.add_argument("--model", default=
                   "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf")
    p.add_argument("--h_epochs", type=int, default=8)
    p.add_argument("--s_epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--out_json", default="runs/terrain_da_ft.json")
    p.add_argument("--net_out", default="runs/terrain_da_ft.pt")
    args = p.parse_args()

    dev = "cuda"
    files = sorted(glob.glob(os.path.join(args.data, "*.npz")))
    hold = max(3, len(files) // 5)
    tr_f, te_f = files[:-hold], files[-hold:]   # 留出集与 Y2 系恒同
    if args.extra_data:
        tr_f = tr_f + sorted(glob.glob(os.path.join(args.extra_data, "*.npz")))
    TR, TE = load_frames(tr_f), load_frames(te_f)
    print(f"[y2f] train {len(TR)} / holdout {len(TE)} 帧", flush=True)

    net = DAHeightFT(args.model).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(0)
    for e in range(args.h_epochs):
        idx = rng.permutation(len(TR))
        tot = 0.0
        net.train()
        for i0 in range(0, len(idx), 4):
            bi = idx[i0:i0 + 4]
            rgb = torch.stack([torch.from_numpy(TR[i][0]) for i in bi]).to(dev)
            pit = torch.tensor([TR[i][1] for i in bi], device=dev)
            tgt = torch.stack([torch.from_numpy(TR[i][2]) for i in bi]
                              )[:, None].float().to(dev)
            loss = F.huber_loss(net(rgb, pit), tgt, delta=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(bi)
        print(f"[y2f] ft epoch {e} loss={tot/len(idx):.5f}", flush=True)
    torch.save(net.state_dict(), args.net_out)
    net.eval()

    @torch.no_grad()
    def pred_h(s):
        rgb = torch.from_numpy(s[0])[None].to(dev)
        pit = torch.tensor([s[1]], device=dev)
        return net(rgb, pit)[0].cpu()

    errs = [float((pred_h(s)[0] - torch.from_numpy(s[2])).abs().mean())
            for s in TE]
    h_mae = float(np.mean(errs))
    print(f"[y2f] holdout 高度 MAE={h_mae:.4f}", flush=True)

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
        print(f"[y2f] seg epoch {e} loss={tot/len(idx):.4f}", flush=True)

    ious = {c: [] for c in TCLASSES}
    with torch.no_grad():
        for s in TE:
            lab = head(pred_h(s)[None].float().to(dev))[0].argmax(0
                                                                  ).cpu().numpy()
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
               model=args.model, inputs="RGB+pitch,零特权(监督=解析高度)")
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[y2f] {json.dumps(out, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
