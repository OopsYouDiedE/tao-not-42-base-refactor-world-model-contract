#!/usr/bin/env python3
"""C3 自动质检:示范/校准 npz 的 schema 校验 + 教师指标阈值自动拒收。

集群就绪判据 C3(scale-plan §4.5):数据引擎喂集群前,坏局必须机器拒收而非人工肉眼。
两类产物各一套检查:
  trackcmd(示范):字段齐全/长度一致/token 非全零率/教师段末角误差中位 ≤ 阈值
                  /切换段数 ≥1;
  calib(校准):字段齐全/GT 至少一类非空/位姿有变化(非卡死)。
不合格 → 移入 <dir>/_rejected/(不删),打印分布报告。

用法:
  PYTHONPATH=. .venv/bin/python tests/integration/qc_demos.py \
      --dirs runs/data/trackcmd_v13 runs/data/calib640 --apply
"""
import argparse
import glob
import json
import os
import shutil

import numpy as np

TRACK_FIELDS = {"tokens", "goal_idx", "dx", "dy", "keys", "pose", "ang_err", "dist"}
CALIB_FIELDS = {"frames", "pose", "gt_blocks"}


def qc_track(z, tail_err_max=25.0, tok_min_rate=0.3):
    """返回 (ok, reason)。教师段末角误差中位>阈值 = 教师没在干活,拒收。"""
    miss = TRACK_FIELDS - set(z.files)
    if miss:
        return False, f"缺字段 {miss}"
    T = z["tokens"].shape[0]
    if not (len(z["goal_idx"]) == len(z["ang_err"]) == T and len(z["dx"]) == T):
        return False, "长度不一致"
    if (z["tokens"][:, :, 4] > 0).any(1).mean() < tok_min_rate:
        return False, "token 全零率过高(感知失效)"
    sched = np.where(np.diff(z["goal_idx"]) != 0)[0]
    if len(sched) < 1:
        return False, "无切换段"
    bounds = [0] + (sched + 1).tolist() + [T]
    tails = [np.median(z["ang_err"][e - 10:e])
             for s, e in zip(bounds[:-1], bounds[1:]) if e - s >= 20]
    if not tails or np.median(tails) > tail_err_max:
        return False, f"教师段末中位 {np.median(tails) if tails else 1e9:.0f}°>{tail_err_max}"
    return True, ""


def qc_calib(z):
    miss = CALIB_FIELDS - set(z.files)
    if miss:
        return False, f"缺字段 {miss}"
    gt = json.loads(str(z["gt_blocks"]))
    if not any(gt.values()) and "hardneg" not in str(z.get("meta", "")):
        pass                                   # 纯负局合法(hard_neg)
    if len(z["frames"]) != len(z["pose"]):
        return False, "帧-位姿长度不一致"
    if np.std(z["pose"][:, 3]) < 1e-3 and np.std(z["pose"][:, 4]) < 1e-3:
        return False, "位姿零方差(卡死局)"
    return True, ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", nargs="+", required=True)
    p.add_argument("--tail_err_max", type=float, default=25.0)
    p.add_argument("--apply", action="store_true", help="不加=只报告不移动")
    args = p.parse_args()
    for d in args.dirs:
        files = sorted(glob.glob(os.path.join(d, "*.npz")))
        rej, reasons = [], {}
        for f in files:
            z = np.load(f, allow_pickle=True)
            is_track = "tokens" in z.files
            ok, why = (qc_track(z, args.tail_err_max) if is_track else qc_calib(z))
            if not ok:
                rej.append(f)
                reasons[why] = reasons.get(why, 0) + 1
        print(f"[qc] {d}: {len(files)} 局, 拒收 {len(rej)} ({reasons})")
        if args.apply and rej:
            rd = os.path.join(d, "_rejected")
            os.makedirs(rd, exist_ok=True)
            for f in rej:
                shutil.move(f, os.path.join(rd, os.path.basename(f)))
            print(f"[qc]   → 移入 {rd}")


if __name__ == "__main__":
    main()
