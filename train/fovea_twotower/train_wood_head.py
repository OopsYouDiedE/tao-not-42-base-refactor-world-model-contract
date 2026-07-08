#!/usr/bin/env python3
"""v7 木类 4 类头训练(WOOD_CLASSES;配方=v4 课程 iron/coal/dirt 原班 + 自然 log)。

扩类标准流程(R-A 定标,每个新类照此办理):
- 课程正样本 = v4 原班目录(calib640{,_rand,_rand2} + hardneg 纯石墙负 +
  trackcmd_motion_frames 闭环运动帧)——iron/coal/dirt 训练数据与 v4 逐像素一致,
  保证扩 log 后课程域回归口径干净(见 eval_wood_head 的 G-W3 门);
- log 正样本 = wood_gt(自然橡树 raycast 扫描 GT,8 角凸包投影);
- log 负样本 = wood_negcert(认证无树自然负帧)。calib_nat_neg 含未标注树对 log
  有毒(已登记),排除;calib_nat(自然铁)未纳入 v4 iron 口径,一并排除保回归纯净。
- neg_frac=0.35(沿 v4;纯负帧按比例掺,不纯负轰炸,治铁类假阳性)。

留出:calib640_rand3(课程回归)+ wood_gt_hold(log)由 eval_wood_head 消费,
本脚本不训练它们。

用法:
  PYTHONPATH=. .venv/bin/python train/fovea_twotower/train_wood_head.py \
      --out runs/g1_conv_head_v7_wood.pt
"""
import argparse

import numpy as np
import torch

from net.fovea_twotower.seg_head import ConvSegHead
from net.fovea_twotower.wood import WOOD_CLASSES, wood_label_img
from net.fovea_twotower.yolo_unified import UnifiedYoloe26, pad384
from train.fovea_twotower.eval_g1 import load_eps

DIRS_DEFAULT = ["runs/data/calib640", "runs/data/calib640_rand",
                "runs/data/calib640_rand2", "runs/data/calib640_hardneg",
                "runs/data/trackcmd_motion_frames",
                "runs/data/wood_gt", "runs/data/wood_negcert"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dirs", nargs="+", default=DIRS_DEFAULT)
    p.add_argument("--out", default="runs/g1_conv_head_v7_wood.pt")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--neg_frac", type=float, default=0.35)
    args = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    u = UnifiedYoloe26(device=dev)
    C = len(WOOD_CLASSES)
    feats, labs = [], []
    n_pos = n_neg = 0
    rng = np.random.default_rng(0)
    for d in args.data_dirs:
        try:
            eps = load_eps(d)
        except AssertionError:
            print(f"[v6] 空目录跳过 {d}")
            continue
        for ep in eps:
            for t in range(0, len(ep["frames"]), 2):
                img = pad384(ep["frames"][t].transpose(1, 2, 0))
                lab = wood_label_img(ep["gt"], ep["pose"][t])
                if (lab != C).sum() < 100:
                    if rng.random() > args.neg_frac or n_neg >= max(n_pos, 40):
                        continue
                    n_neg += 1
                else:
                    n_pos += 1
                feats.append(u.embed(img)[0][0].half().cpu())
                labs.append(torch.from_numpy(lab))
    print(f"[v6] 缓存 {len(feats)} 帧(正 {n_pos}/纯负 {n_neg})", flush=True)
    cnt = torch.stack([(torch.stack(labs) == k).sum() for k in range(C + 1)])
    wgt = (cnt.sum() / (cnt.float() + 1)).sqrt()
    wgt = (wgt / wgt.mean()).to(dev)
    head = ConvSegHead(ncls=C + 1).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)
    for e in range(args.epochs):
        idx = rng.permutation(len(feats))
        tot = 0.0
        for i0 in range(0, len(idx), 8):
            bi = idx[i0:i0 + 8]
            x = torch.stack([feats[i] for i in bi]).float().to(dev)
            y = torch.stack([labs[i] for i in bi]).to(dev)
            loss = torch.nn.functional.cross_entropy(head(x), y, weight=wgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(bi)
        print(f"[v6] epoch {e} loss={tot / len(idx):.4f}", flush=True)
    torch.save(head.state_dict(), args.out)
    print(f"[v6] saved → {args.out}")


if __name__ == "__main__":
    main()
