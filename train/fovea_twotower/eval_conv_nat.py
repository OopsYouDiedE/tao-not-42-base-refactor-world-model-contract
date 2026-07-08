#!/usr/bin/env python3
"""感知自然域校准三门评估(v5 预登记门,docs/architectures/fovea-experiments-index.md)。

G-N1 留出纯自然负帧铁假阳性 <20px/帧;G-N2 留出自然正样本铁 IoU ≥0.4;
G-N3 课程域回归:calib640_rand3 mIoU 相对基线头降幅 ≤10%。
双头并报(基线 v4 vs 候选 v5),回归口径=同脚本同帧集。

用法:
  PYTHONPATH=. .venv/bin/python train/fovea_twotower/eval_conv_nat.py \
      --cand runs/g1_conv_head_v5.pt --base runs/g1_conv_head_v4.pt
"""
import argparse
import json

import numpy as np
import torch

from net.fovea_twotower.seg_head import ConvSegHead
from net.fovea_twotower.token_stream import CLASSES
from net.fovea_twotower.yolo_unified import UnifiedYoloe26, pad384
from train.fovea_twotower.eval_g1 import (gt_masks, iou, load_eps,
                                          pred_mask_conv)


def load_head(path, dev):
    h = ConvSegHead().to(dev).eval()
    h.load_state_dict(torch.load(path, map_location=dev, weights_only=False))
    return h


def iron_fp_px(u, head, eps, stride=3):
    px = []
    for ep in eps:
        for t in range(0, len(ep["frames"]), stride):
            img = pad384(ep["frames"][t].transpose(1, 2, 0))
            pred = pred_mask_conv(u, img, head, tta=True)
            px.append(int(pred["iron_ore"].sum()))
    return float(np.mean(px)), float(np.median(px))


def iron_iou(u, head, eps, stride=3, min_gt_px=250):
    vals = []
    for ep in eps:
        for t in range(0, len(ep["frames"]), stride):
            img = pad384(ep["frames"][t].transpose(1, 2, 0))
            gt = gt_masks(ep["gt"], ep["pose"][t])
            if gt["iron_ore"].sum() < min_gt_px:
                continue
            pred = pred_mask_conv(u, img, head, tta=True)
            v = iou(pred["iron_ore"], gt["iron_ore"])
            if v is not None:
                vals.append(v)
    return float(np.mean(vals)) if vals else None, len(vals)


def course_miou(u, head, eps, stride=3, min_gt_px=250):
    ious = {c: [] for c in CLASSES}
    for ep in eps:
        for t in range(0, len(ep["frames"]), stride):
            img = pad384(ep["frames"][t].transpose(1, 2, 0))
            gt = gt_masks(ep["gt"], ep["pose"][t])
            pred = pred_mask_conv(u, img, head, tta=True)
            for c in CLASSES:
                if gt[c].sum() >= min_gt_px:
                    v = iou(pred[c], gt[c])
                    if v is not None:
                        ious[c].append(v)
    per = {c: float(np.mean(v)) if v else None for c, v in ious.items()}
    vals = [v for v in per.values() if v is not None]
    return float(np.mean(vals)), per


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cand", required=True)
    p.add_argument("--base", default="runs/g1_conv_head_v4.pt")
    p.add_argument("--neg_hold", default="runs/data/calib_nat_neg_hold")
    p.add_argument("--pos_hold", default="runs/data/calib_nat_hold")
    p.add_argument("--course_hold", default="runs/data/calib640_rand3")
    p.add_argument("--out_json", default="runs/conv_nat_gates.json")
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    u = UnifiedYoloe26(device=dev)
    neg = load_eps(args.neg_hold)
    pos = load_eps(args.pos_hold)
    course = load_eps(args.course_hold)
    out = {}
    for tag, path in (("base", args.base), ("cand", args.cand)):
        head = load_head(path, dev)
        fp_mean, fp_med = iron_fp_px(u, head, neg)
        iiou, n_iou = iron_iou(u, head, pos)
        cm, per = course_miou(u, head, course)
        out[tag] = dict(path=path, iron_fp_px_mean=round(fp_mean, 1),
                        iron_fp_px_med=round(fp_med, 1),
                        nat_iron_iou=round(iiou, 3) if iiou is not None else None,
                        n_iou_frames=n_iou, course_miou=round(cm, 3),
                        course_per={k: round(v, 3) if v else None
                                    for k, v in per.items()})
        print(f"[{tag}] {json.dumps(out[tag], ensure_ascii=False)}", flush=True)
    c, b = out["cand"], out["base"]
    gates = {
        "G-N1(负帧铁FP<20px/帧)": bool(c["iron_fp_px_mean"] < 20),
        "G-N2(自然铁IoU>=0.4)": bool((c["nat_iron_iou"] or 0) >= 0.4),
        "G-N3(课程域降幅<=10%)": bool(c["course_miou"] >= 0.9 * b["course_miou"]),
    }
    out["gates"] = gates
    print(json.dumps(gates, ensure_ascii=False, indent=2))
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
