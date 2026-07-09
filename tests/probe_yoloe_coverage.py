#!/usr/bin/env python3
"""YOLOE 提案覆盖率体检:K×7 token 到底把世界删掉了多少?

背景(2026-07-09 用户裁决:去掉 n_cls;并按苦涩的教训审查表征):
  现行 `net/fovea_twotower/yolo_parse.py` 给快塔的观测是
      tokens [B, K=8, PARSE_DIM=7] = [cls/n_cls, cx, cy, w, h, conf, area]
  类别只能取自 `wood.py::WOOD_CLASSES = ["iron_ore","coal_ore","dirt","log"]`。
  实测(森林世界 48 次 raycast):**77% 挡住准星的方块不在该词表里**,其中 11 次是树叶
  —— 正是 R-A 归因 latch=0 的那个"叶冠挡 raycast"。

本探针分离两个独立的信息瓶颈:
  (a) **词表瓶颈**:promptable 通路只认 4 个类;
  (b) **框瓶颈**:prompt-free 通路虽无词表,但仍把世界分解成 K 个框/掩膜。

判据(逐帧,客观):
  - 各通路的提案掩膜并集覆盖了画面多少像素;
  - **准星像素(画面正中)是否被任一提案掩膜覆盖** —— 这是 raycast 真正撞上的东西;
  - top-K 截断后还剩多少覆盖率。

去掉 n_cls 只解决 (a)。若 pf 通路的准星覆盖率也低,则 (b) 同样要动。

用法:
    /workspace/venv-mc/bin/python tests/probe_yoloe_coverage.py \
        --frames "/workspace/assets/forest_s*.png"
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import torch
import torchvision
from PIL import Image


def patch_nms_to_cpu() -> None:
    """torchvision 0.26.0+cu129 的 _C.so 只含 sm_50..sm_90 的 cubin,**没有 sm_120**。

    在 RTX 5090 上任何 torchvision CUDA 自定义算子(nms/roi_align/deform_conv)都会抛
        torch.AcceleratorError: CUDA error: no kernel image is available for execution
    且无可用 PTX 回退。ultralytics 的 NMS 直接调 torchvision.ops.nms ⇒ YOLOE 在 5090 上
    开箱即挂。框数只有几百,搬到 CPU 的代价可忽略。

    验证:cuobjdump --list-elf torchvision/_C*.so  →  sm_50 60 70 75 80 86 90
    """
    _orig = torchvision.ops.nms

    def _nms(boxes, scores, iou):
        if boxes.is_cuda:
            keep = _orig(boxes.cpu(), scores.cpu(), iou)
            return keep.to(boxes.device)
        return _orig(boxes, scores, iou)

    torchvision.ops.nms = _nms
    torchvision.ops.boxes.nms = _nms
    # ultralytics 在调用点 `torchvision.ops.nms(...)` 动态解析,故上面两行已足够。
    # (cu130 的 torchvision 预计带 sm_120 cubin,届时本补丁可删。)

PF_W = "runs/checkpoints/yoloe-11l-seg-pf.pt"
PROMPT_W = "runs/checkpoints/yoloe-11l-seg.pt"
WOOD_CLASSES = ["iron_ore", "coal_ore", "dirt", "log"]   # net/fovea_twotower/wood.py
K = 8            # yolo_parse.YoloParseHead 默认截断(要证伪的就是它)
MAX_DET = 256    # YOLOE pf 端到端提案上限:提示向量只负责"从这 256 个里挑",不负责感知


def union_mask(masks, hw) -> np.ndarray:
    u = np.zeros(hw, bool)
    for m in masks:
        u |= m
    return u


def to_full(masks_data, hw) -> list[np.ndarray]:
    out = []
    for m in masks_data:
        a = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize(
            (hw[1], hw[0]), Image.NEAREST)) > 127
        out.append(a)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="/workspace/assets/forest_s*.png")
    ap.add_argument("--conf-pf", type=float, default=0.01)
    ap.add_argument("--conf-prompt", type=float, default=0.02)
    args = ap.parse_args()

    patch_nms_to_cpu()
    from ultralytics import YOLOE

    paths = sorted(glob.glob(args.frames))
    if not paths:
        raise SystemExit(f"no frames match {args.frames}")

    pf = YOLOE(PF_W)
    pm = YOLOE(PROMPT_W)
    pm.set_classes(WOOD_CLASSES, pm.get_text_pe(WOOD_CLASSES))

    rows = []
    for p in paths:
        img = np.asarray(Image.open(p).convert("RGB"))
        H, W = img.shape[:2]
        cy, cx = H // 2, W // 2                       # 准星像素

        rec = {"frame": p.split("/")[-1]}
        for tag, model, conf in (("pf(无词表)", pf, args.conf_pf),
                                 ("prompt(4类)", pm, args.conf_prompt)):
            r = model.predict(img, imgsz=(384, 640), conf=conf, max_det=MAX_DET,
                              verbose=False, device="cuda")[0]
            n = 0 if r.boxes is None else len(r.boxes)
            if n == 0 or r.masks is None:
                rec[tag] = dict(n=n, cover=0.0, cross=False, cover_topk=0.0, cross_topk=False)
                continue
            confs = r.boxes.conf.cpu().numpy()
            masks = to_full(r.masks.data.cpu().numpy() > 0.5, (H, W))
            u = union_mask(masks, (H, W))
            order = np.argsort(-confs)[:K]
            uk = union_mask([masks[i] for i in order], (H, W))
            rec[tag] = dict(n=n, cover=float(u.mean()), cross=bool(u[cy, cx]),
                            cover_topk=float(uk.mean()), cross_topk=bool(uk[cy, cx]))
        rows.append(rec)

    print(f"{'frame':22s} | {'pf n':>5} {'cover':>6} {'cross':>5} {'topK cov':>8} {'topK cross':>10}"
          f" || {'pm n':>4} {'cover':>6} {'cross':>5}")
    print("-" * 100)
    for r in rows:
        a, b = r["pf(无词表)"], r["prompt(4类)"]
        print(f"{r['frame']:22s} | {a['n']:5d} {a['cover']:6.1%} {str(a['cross']):>5} "
              f"{a['cover_topk']:8.1%} {str(a['cross_topk']):>10} || "
              f"{b['n']:4d} {b['cover']:6.1%} {str(b['cross']):>5}")

    def agg(tag, key):
        return float(np.mean([r[tag][key] for r in rows]))

    print("\n=== 汇总 ===")
    print(f"pf(无词表)   : 平均提案 {agg('pf(无词表)', 'n'):.1f} 个, "
          f"像素覆盖 {agg('pf(无词表)', 'cover'):.1%}, 准星被覆盖 {agg('pf(无词表)', 'cross'):.0%} 的帧")
    print(f"  截断到 top-{K}: 覆盖 {agg('pf(无词表)', 'cover_topk'):.1%}, "
          f"准星被覆盖 {agg('pf(无词表)', 'cross_topk'):.0%} 的帧")
    print(f"prompt(4类) : 平均检出 {agg('prompt(4类)', 'n'):.1f} 个, "
          f"像素覆盖 {agg('prompt(4类)', 'cover'):.1%}, 准星被覆盖 {agg('prompt(4类)', 'cross'):.0%} 的帧")
    print("\n读法:准星覆盖率 = 快塔能否'看见'挡住自己 raycast 的那个东西。")
    print("     去掉 n_cls 只把 prompt 行换成 pf 行;若 pf 行的准星覆盖率也低,")
    print("     则 K 个框这个分解本身在删信息,要连框一起换成稠密特征。")


if __name__ == "__main__":
    main()
