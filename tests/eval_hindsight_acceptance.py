#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""hindsight relabel BC 的双对照验收（见 knowledge/README.md §2.2；对照臂纪律，不许跳）。

两道门,全过才算"goal 通道修通":
  1. holdout 有标签 tick(GUI 剔除)上,真 goal 的 ce+bce **显著低于**组内 permute
     goal(clip 内有标签 tick 之间打乱——边缘分布不变、对齐被破坏;配对差值
     bootstrap 95% CI 上界 < 0 判显著)。
  2. zero-goal 全 tick 口径不明显劣于 canonical(bc_vpt/best.pt,同 holdout 同代码
     重测,不信档案数字)——阈值:劣化 ≤ 2%(相对)。

用法:python -m tests.eval_hindsight_acceptance \
          --ckpt runs/checkpoints/bc_vpt4/best.pt [--canonical runs/checkpoints/bc_vpt/best.pt]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from net.pixel_tower import PixelTowerConfig, build_pixel_tower  # noqa: E402
from train.craftground.bc_vpt_warmstart import evaluate, load_holdout  # noqa: E402


def _load_tower(path: str, device: str):
    ck = torch.load(path, map_location=device, weights_only=True)
    cfg = PixelTowerConfig(**{k: (tuple(v) if isinstance(v, list) else v)
                              for k, v in ck["cfg"].items()})
    tower = build_pixel_tower(cfg).to(device)
    tower.load_state_dict(ck["tower"])
    return tower, ck.get("step")


def paired_bootstrap(diff: np.ndarray, n_boot: int = 10000, seed: int = 0):
    """配对差值均值的 bootstrap 95% CI。diff = per_tick(true) - per_tick(perm)。"""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), (n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), \
        float(np.percentile(means, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/checkpoints/bc_vpt4/best.pt")
    ap.add_argument("--canonical", default="runs/checkpoints/bc_vpt/best.pt")
    ap.add_argument("--holdout", default="runs/data/vpt_holdout")
    ap.add_argument("--goal-vocab", default="runs/data/vpt_early_goal_vocab.json")
    ap.add_argument("--perm-seeds", type=int, default=5, help="permute 臂重复次数(取均值)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tower, step = _load_tower(args.ckpt, device)
    clips = load_holdout(args.holdout, tower.cfg.frame_stack, goal_vocab=args.goal_vocab)
    n_lab = int(sum((c["ok"] & c["labeled"]).sum() for c in clips))
    print(f"holdout 有标签 tick(GUI 剔除)= {n_lab}", flush=True)

    # 门 1:有标签 tick 上 真 goal vs 组内 permute(配对,多 seed)
    lt = evaluate(tower, clips, device, goal_mode="true", labeled_only=True,
                  per_tick=True)
    true_score = lt["ce"] + lt["bce"]
    perm_scores, diffs = [], []
    for s in range(args.perm_seeds):
        lp = evaluate(tower, clips, device, goal_mode="perm", labeled_only=True,
                      perm_seed=s, per_tick=True)
        perm_scores.append(lp["ce"] + lp["bce"])
        diffs.append(lt["per_tick"] - lp["per_tick"])
    diff = np.concatenate(diffs)
    mean_d, lo, hi = paired_bootstrap(diff)
    gate1 = hi < 0.0
    print(f"[门1] labeled true={true_score:.4f} perm={np.mean(perm_scores):.4f} "
          f"(seeds={args.perm_seeds}) 配对差 {mean_d:+.4f} CI95 [{lo:+.4f},{hi:+.4f}] "
          f"→ {'PASS' if gate1 else 'FAIL'}", flush=True)

    # 门 2:zero-goal 全 tick vs canonical 同代码重测
    mz = evaluate(tower, clips, device, goal_mode="zero")
    zero_score = mz["ce"] + mz["bce"]
    can_tower, can_step = _load_tower(args.canonical, device)
    mc = evaluate(can_tower, clips, device, goal_mode="zero")
    can_score = mc["ce"] + mc["bce"]
    gate2 = zero_score <= can_score * 1.02
    print(f"[门2] zero-goal {zero_score:.4f} vs canonical {can_score:.4f}"
          f"(step={can_step}) 相对 {100*(zero_score/can_score-1):+.2f}% "
          f"→ {'PASS' if gate2 else 'FAIL'}", flush=True)

    out = dict(ckpt=args.ckpt, step=step, n_labeled_tick=n_lab,
               lab_true=round(true_score, 4),
               lab_perm=round(float(np.mean(perm_scores)), 4),
               paired_diff=round(mean_d, 4), ci95=[round(lo, 4), round(hi, 4)],
               zero=round(zero_score, 4), canonical_zero=round(can_score, 4),
               zero_cam_acc=mz["cam_acc"], zero_key_f1=mz["key_f1_mean"],
               true_cam_acc=lt["cam_acc"], true_key_f1=lt["key_f1_mean"],
               gate1_true_beats_perm=bool(gate1), gate2_zero_not_worse=bool(gate2),
               verdict="PASS" if (gate1 and gate2) else "FAIL")
    p = Path(args.ckpt).parent / "acceptance.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"→ {p}\n{json.dumps(out, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
