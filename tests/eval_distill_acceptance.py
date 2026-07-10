#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VPT 蒸馏专项验收(验收阶梯②;①双门走 eval_hindsight_acceptance,③走 grpo --smoke)。

同 holdout 同代码重测两个 checkpoint(蒸馏学生 vs 基线 bc_vpt4),报告:
  1. 学生 vs 教师动作一致率:键(sigmoid>0.5 对 教师 p>0.5 的逐键一致率均值 +
     教师作标签的 attack F1)与相机(argmax bin 对 教师 remap 后 argmax bin 命中率);
  2. attack 键对人类真值的 P/R/F1(evaluate() 原口径,канonical zero-goal);
  3. 通用 BC 指标回归对照(ce+bce zero/true 口径,防蒸馏把 goal 通道压聋的旁证——
     正门仍是①的双门)。

教师标签来自 train/minecraft/vpt_teacher.py 打标的 npz(holdout 目录也要打)。

用法:python -m tests.eval_distill_acceptance \
        --ckpt runs/checkpoints/bc_distill1_w05/best.pt --baseline runs/checkpoints/bc_vpt4/best.pt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from net.pixel_tower import PixelTowerConfig, build_pixel_tower  # noqa: E402
from train.craftground.action_contract import CAM_BINS, V2_KEYS  # noqa: E402
from train.craftground.bc_vpt_warmstart import (CAMERA_SCALE, IMG_HW, evaluate,  # noqa: E402
                                                load_holdout)
from train.minecraft.vpt_dataset import VPTStreamDataset  # noqa: E402
from train.minecraft.vpt_teacher import remap_cam  # noqa: E402


def _load_tower(path: str, device: str):
    ck = torch.load(path, map_location=device, weights_only=True)
    cfg = PixelTowerConfig(**{k: (tuple(v) if isinstance(v, list) else v)
                              for k, v in ck["cfg"].items()})
    tower = build_pixel_tower(cfg).to(device)
    tower.load_state_dict(ck["tower"])
    tower.eval()
    return tower, ck.get("step")


def load_teacher_holdout(holdout_dir: str, labels_dir: str, s: int) -> list[dict]:
    """holdout 段 + 教师标签(与 load_holdout 同帧堆叠/GUI 口径,只保留有教师标签段)。"""
    from train.craftground.action_contract import stack_frames
    ds = VPTStreamDataset(holdout_dir, seq_len=8, img_size=IMG_HW,
                          camera_scale=CAMERA_SCALE, split=None, teacher_dir=labels_dir)
    clips = []
    for mp4, jsonl in ds.pairs:
        c = ds._load_clip(mp4, jsonl)
        if c is None or "tch_keys" not in c:
            continue
        imgs = c["img"].numpy().transpose(0, 2, 3, 1)
        gui = c["gui"]
        ok = np.ones(c["n"], bool) if gui is None else (gui.numpy() < 0.5)
        ok[:s - 1] = False
        ok &= c["tch_on"].numpy()
        clips.append(dict(stacked=stack_frames(imgs, s), ok=ok, n=c["n"],
                          tch_keys=c["tch_keys"].float(),
                          tch_cam=remap_cam(c["tch_cam"].float())))
    if not clips:
        raise RuntimeError(f"{holdout_dir} 无带教师标签的段——先跑 vpt_teacher 打标(含 holdout)")
    return clips


@torch.no_grad()
def teacher_agreement(tower, clips: list[dict], device, chunk: int = 512) -> dict:
    """学生(zero-goal 部署口径)vs 教师:键一致率 / 教师标签 attack F1 / 相机 bin 命中率。"""
    n = 0
    key_agree = np.zeros(len(V2_KEYS))
    cam_hit = np.zeros(2)
    a_i = V2_KEYS.index("attack")
    tp = fp = fn = 0
    kl_sum = 0.0
    for clip in clips:
        for i0 in range(0, clip["n"], chunk):
            sl = slice(i0, min(i0 + chunk, clip["n"]))
            m = clip["ok"][sl]
            if not m.any():
                continue
            img = torch.from_numpy(clip["stacked"][sl][m]).float().div_(255.0)
            img = img.unsqueeze(1).to(device)
            goal = torch.zeros(img.shape[0], 384 + 2, device=device)
            prev = torch.zeros(img.shape[0], 1, 22, device=device)
            cam_l, key_l = tower(img, goal, prev)
            cam_l = cam_l[:, 0, 0].float()                       # [n,2,11]
            key_l = key_l[:, 0, 0].float()                       # [n,20]
            tk = clip["tch_keys"][sl][m].to(device)              # [n,20] 概率
            tc = clip["tch_cam"][sl][m].to(device)               # [n,2,11] 我们 bin
            sp = (torch.sigmoid(key_l) > 0.5)
            tpred = tk > 0.5
            key_agree += (sp == tpred).float().sum(0).cpu().numpy()
            cam_hit += (cam_l.argmax(-1) == tc.argmax(-1)).float().sum(0).cpu().numpy()
            tp += int((sp[:, a_i] & tpred[:, a_i]).sum())
            fp += int((sp[:, a_i] & ~tpred[:, a_i]).sum())
            fn += int((~sp[:, a_i] & tpred[:, a_i]).sum())
            q = F.log_softmax(cam_l, -1)
            p = tc.clamp(min=1e-4); p = (p / p.sum(-1, keepdim=True)).log()
            kl_sum += float((q.exp() * (q - p)).sum(-1).mean(-1).sum())
            n += img.shape[0]
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    per_key = {k: round(float(key_agree[i] / max(n, 1)), 4) for i, k in enumerate(V2_KEYS)}
    return dict(n_tick=n,
                key_agree_mean=round(float(key_agree.mean() / max(n, 1)), 4),
                attack_agree=per_key["attack"],
                attack_f1_vs_teacher=round(2 * prec * rec / max(prec + rec, 1e-4), 4),
                cam_hit_dx=round(float(cam_hit[0] / max(n, 1)), 4),
                cam_hit_dy=round(float(cam_hit[1] / max(n, 1)), 4),
                kl_cam_vs_teacher=round(kl_sum / max(n, 1), 4),
                per_key_agree=per_key)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="蒸馏学生 checkpoint")
    ap.add_argument("--baseline", default="runs/checkpoints/bc_vpt4/best.pt")
    ap.add_argument("--holdout", default="runs/data/vpt_holdout")
    ap.add_argument("--labels", default="runs/data/vpt_labels")
    ap.add_argument("--goal-vocab", default="runs/data/vpt_early_goal_vocab.json")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    report = {}
    tower_s, step_s = _load_tower(args.ckpt, device)
    tclips = load_teacher_holdout(args.holdout, args.labels, tower_s.cfg.frame_stack)
    hclips = load_holdout(args.holdout, tower_s.cfg.frame_stack, goal_vocab=args.goal_vocab)
    for name, path in (("student", args.ckpt), ("baseline", args.baseline)):
        tower, step = _load_tower(path, device)
        agree = teacher_agreement(tower, tclips, device)
        m0 = evaluate(tower, hclips, device, goal_mode="zero")
        mt = evaluate(tower, hclips, device, goal_mode="true")
        report[name] = dict(ckpt=path, step=step, teacher_agreement=agree,
                            zero=dict(score=round(m0["ce"] + m0["bce"], 4),
                                      cam_acc=m0["cam_acc"], key_f1=m0["key_f1_mean"],
                                      attack=m0["per_key"]["attack"]),
                            true=dict(score=round(mt["ce"] + mt["bce"], 4),
                                      attack=mt["per_key"]["attack"]))
        a = report[name]
        print(f"[{name}] step={step} 键一致率={agree['key_agree_mean']} "
              f"attack一致={agree['attack_agree']} attackF1(教师)={agree['attack_f1_vs_teacher']} "
              f"cam命中 dx/dy={agree['cam_hit_dx']}/{agree['cam_hit_dy']} | "
              f"人类真值 attack F1 zero={a['zero']['attack']['f1']} "
              f"zero score={a['zero']['score']} true score={a['true']['score']}", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1))
        print(f"报告 → {args.out}")


if __name__ == "__main__":
    main()
