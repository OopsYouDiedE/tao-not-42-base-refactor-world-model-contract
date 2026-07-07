#!/usr/bin/env python3
"""ConvSegHead 训练(引擎 B 正式脚本,固化 g1_conv_head_v4 配方)。

原散在 scratchpad 的 conv_v2/v3/v4 一次性脚本,固化为可复现入口。配方(v4):
  数据 = 固定布局 + 随机布局 + 纯石墙负样本(--hard_neg) + 闭环运动帧
         (collect_track_cmd --store_frames 产物,"感知 DAgger":在闭环访问分布上训感知,
         教师锁定率 0.28→0.56 的关键杠杆);
  epochs 10 / neg_frac 0.35(纯负帧按比例掺入,专治铁类假阳性:177px/帧→0);
  回归检查 = 留出目录逐类 mIoU(tta),防新数据把旧分布训退。

用法(复现 v4):
  PYTHONPATH=. .venv/bin/python train/fovea_twotower/train_conv_head.py \
      --data_dirs runs/data/calib640 runs/data/calib640_rand runs/data/calib640_rand2 \
                  runs/data/calib640_hardneg runs/data/trackcmd_motion_frames \
      --test_dir runs/data/calib640_rand3 --out runs/g1_conv_head_v4.pt
"""
import argparse

import numpy as np
import torch

from net.fovea_twotower.token_stream import CLASSES
from net.fovea_twotower.yolo_unified import UnifiedYoloe26, pad384
from train.fovea_twotower.eval_g1 import (gt_masks, iou, load_eps,
                                          pred_mask_conv, train_conv_head)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dirs", nargs="+", required=True)
    p.add_argument("--out", default="runs/g1_conv_head.pt")
    p.add_argument("--test_dir", default="", help="留出目录(逐类 mIoU 回归检查)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--neg_frac", type=float, default=0.35)
    p.add_argument("--min_gt_px", type=int, default=250)
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    u = UnifiedYoloe26(device=dev)
    tr = [ep for d in args.data_dirs for ep in load_eps(d)]
    head = train_conv_head(u, tr, epochs=args.epochs, dev=dev,
                           neg_frac=args.neg_frac)
    torch.save(head.state_dict(), args.out)
    print(f"[conv] saved → {args.out}")

    if args.test_dir:
        te = load_eps(args.test_dir)
        ious = {c: [] for c in CLASSES}
        for ep in te:
            for t in range(0, len(ep["frames"]), 3):
                img = pad384(ep["frames"][t].transpose(1, 2, 0))
                gt = gt_masks(ep["gt"], ep["pose"][t])
                pred = pred_mask_conv(u, img, head, tta=True)
                for c in CLASSES:
                    if gt[c].sum() >= args.min_gt_px:
                        v = iou(pred[c], gt[c])
                        if v is not None:
                            ious[c].append(v)
        print(f"[conv] {args.test_dir} 留出 mIoU:",
              {c: round(float(np.mean(v)), 3) if v else None
               for c, v in ious.items()})


if __name__ == "__main__":
    main()
