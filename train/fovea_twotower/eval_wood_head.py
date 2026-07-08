#!/usr/bin/env python3
"""v7 木类 4 类头验收门(R-A 扩类闸门,先于结果登记)。

两门(docs/next_session.md R-A step3):
  G-W1(log 感知):wood_gt_hold 留出 log mIoU ≥ 0.35;
  G-W3(课程回归):calib640_rand3 上 iron/coal/dirt mIoU 相对 v4 基线回退 ≤ 0.03。
候选头(ncls=5,WOOD_CLASSES)与基线 v4(ncls=4,CLASSES)同脚本同帧集对拍,
课程 GT 用前脸投影(gt_masks),log GT 用 wood_masks(前脸课程 + 8 角凸包 log)。

对外接口:main(CLI)。用法:
  PYTHONPATH=. .venv/bin/python train/fovea_twotower/eval_wood_head.py \
      --cand runs/g1_conv_head_v7_wood.pt --base runs/g1_conv_head_v4.pt
"""
import argparse
import json

import numpy as np
import torch

from net.fovea_twotower.seg_head import ConvSegHead, gt_masks
from net.fovea_twotower.token_stream import CLASSES
from net.fovea_twotower.wood import WOOD_CLASSES, wood_masks
from net.fovea_twotower.yolo_unified import UnifiedYoloe26, pad384
from train.fovea_twotower.eval_g1 import iou, load_eps, pred_mask_conv


def load_head(path, ncls, dev):
    h = ConvSegHead(ncls=ncls).to(dev).eval()
    h.load_state_dict(torch.load(path, map_location=dev, weights_only=False))
    return h


def per_class_miou(u, head, eps, classes, gt_fn, want, stride=2, min_gt_px=250):
    """want=评测类列表;gt_fn(gt,pose)→{cls:mask};pred 用 head+classes 口径。"""
    ious = {c: [] for c in want}
    for ep in eps:
        for t in range(0, len(ep["frames"]), stride):
            img = pad384(ep["frames"][t].transpose(1, 2, 0))
            gt = gt_fn(ep["gt"], ep["pose"][t])
            pred = pred_mask_conv(u, img, head, tta=True, classes=classes)
            for c in want:
                if gt[c].sum() >= min_gt_px:
                    v = iou(pred[c], gt[c])
                    if v is not None:
                        ious[c].append(v)
    return {c: (float(np.mean(v)) if v else None, len(v)) for c, v in ious.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cand", default="runs/g1_conv_head_v7_wood.pt")
    p.add_argument("--base", default="runs/g1_conv_head_v4.pt")
    p.add_argument("--log_hold", default="runs/data/wood_gt_hold")
    p.add_argument("--course_hold", default="runs/data/calib640_rand3")
    p.add_argument("--out_json", default="runs/wood_head_gates.json")
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    u = UnifiedYoloe26(device=dev)
    log_eps = load_eps(args.log_hold)
    course_eps = load_eps(args.course_hold)

    cand = load_head(args.cand, len(WOOD_CLASSES) + 1, dev)
    base = load_head(args.base, len(CLASSES) + 1, dev)

    # G-W1: log 感知(候选头)
    log_res = per_class_miou(u, cand, log_eps, WOOD_CLASSES, wood_masks, ["log"])
    log_miou, log_n = log_res["log"]

    # G-W3: 课程 iron/coal/dirt 回归(候选 vs 基线,前脸 GT 同口径)
    c_cand = per_class_miou(u, cand, course_eps, WOOD_CLASSES, gt_masks, CLASSES)
    c_base = per_class_miou(u, base, course_eps, CLASSES, gt_masks, CLASSES)
    mean_cand = float(np.mean([v for v, _ in c_cand.values() if v is not None]))
    mean_base = float(np.mean([v for v, _ in c_base.values() if v is not None]))
    regress = mean_base - mean_cand

    gates = {
        "G-W1(log留出mIoU>=0.35)": bool((log_miou or 0) >= 0.35),
        "G-W3(课程回退<=0.03)": bool(regress <= 0.03),
    }
    out = dict(
        cand=args.cand, base=args.base,
        log_miou=round(log_miou, 3) if log_miou is not None else None,
        log_n_frames=log_n,
        course_mean_cand=round(mean_cand, 3),
        course_mean_base=round(mean_base, 3),
        course_regress=round(regress, 3),
        course_per_cand={c: round(v, 3) if v is not None else None
                         for c, (v, _) in c_cand.items()},
        course_per_base={c: round(v, 3) if v is not None else None
                         for c, (v, _) in c_base.items()},
        gates=gates, verdict="PASS" if all(gates.values()) else "FAIL")
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
