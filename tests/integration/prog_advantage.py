#!/usr/bin/env python3
"""程序化密奖励 → 组内相对优势(GRPO 口径),替代 LLM judge 作可靠对照。

第一步(证闭环):把"文字指示能否操纵执行"这个问题,用**可测**奖励隔离出来——
奖励是轨迹摘要的确定性函数,零判优噪声。组内(同指令同起点)对奖励做 z-score →
优势 advantages.json {traj_id: adv},喂 rest_update 做优势加权 BC。

指令→奖励(密、可测):
  mine  = 3·iron_collected + 2·mined_iron + 1.0·ore_in_crosshair_frac   (挖矿活动+成果)
  still = −swings/steps − (|net_yaw|+|net_pitch|)/180 − forward_steps/steps  (越静越高)
  down  = net_pitch/90                       (备用:看下=俯,未启用)
  left  = −net_yaw/90 ; right = net_yaw/90    (备用,判优方向盲已知不可视,仅程序侧可测)

用法:
  PYTHONPATH=. python tests/integration/prog_advantage.py --round_dir runs/rest_r0 \
      --out runs/rest_r0/advantages.json
"""
import argparse
import json
import os

import numpy as np


def reward(instr, s):
    st = max(s["steps"], 1)
    if instr in ("mine", "approach_mine", "scan_mine"):
        return 3.0 * s["iron_collected"] + 2.0 * s["mined_iron"] + 1.0 * s["ore_in_crosshair_frac"]
    if instr == "still":
        return -(s["swings"] / st) - (abs(s["net_yaw_deg"]) + abs(s["net_pitch_deg"])) / 180.0 \
               - (s["forward_steps"] / st)
    if instr == "down":
        return s["net_pitch_deg"] / 90.0
    if instr == "left":
        return -s["net_yaw_deg"] / 90.0
    if instr == "right":
        return s["net_yaw_deg"] / 90.0
    return 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--round_dir", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    gj = json.load(open(os.path.join(args.round_dir, "groups.json")))
    adv = {}
    print(f"[prog] {args.round_dir}: {len(gj['groups'])} groups")
    for g in gj["groups"]:
        instr = g["instr"]
        rows = [(tr["traj_id"], reward(instr, tr["summary"])) for tr in g["trajectories"]]
        rs = np.array([r for _, r in rows], dtype=np.float64)
        mu, sd = rs.mean(), rs.std()
        for (tid, r) in rows:
            a = (r - mu) / (sd + 1e-6)
            adv[tid] = round(float(np.clip(a, -2.0, 2.0)), 4)
        order = sorted(rows, key=lambda x: -x[1])
        print(f"  {g['group_id']:14s} R[min={rs.min():.2f} mean={mu:.2f} max={rs.max():.2f} sd={sd:.2f}] "
              f"best={order[0][0].split('__')[-1]}({order[0][1]:.2f}) "
              f"worst={order[-1][0].split('__')[-1]}({order[-1][1]:.2f})")
    out = args.out or os.path.join(args.round_dir, "advantages.json")
    json.dump({"advantages": adv, "source": "programmatic"}, open(out, "w"), indent=1, ensure_ascii=False)
    print(f"💾 {out} | {len(adv)} 轨优势(组内 z-score,clip±2)")


if __name__ == "__main__":
    main()
