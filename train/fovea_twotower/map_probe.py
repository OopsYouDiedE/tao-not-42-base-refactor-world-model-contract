#!/usr/bin/env python3
"""M3-pre 地图机制探针:北锚定防糊 + clipmap 分级精度,合成世界三臂对决。

问题(用户 2026-07-08):自我中心地图结构能否对不同远近给出不同精度?
以及一个更基础的决策:旋转怎么处理——逐步重采样(文献默认)还是北锚定记账?

任务:2D 连续世界随机地标(c=8 随机特征),agent 随机游走 T 步,观测半径内地标
逐步写图;结束时按"当前距离"分桶考读出特征与真值的余弦相似度。
三臂(唯一差异=地图结构,写入流完全相同):
  naive  均匀 64×64,每步旋转+平移重采样(4096 cell)
  north  均匀 64×64,北锚定(整格roll+亚格寄存器,旋转推迟读出)(4096 cell)
  clip   北锚定 clipmap 3 级 32×32,近环精(0.5u/cell)远环粗,半径 2×(3072 cell,75%预算)

门(先登记):
  G-north: north 近场(r<8)相似度 ≥ naive + 0.05  → 逐步重采样=糊的主因,北锚定成立
  G-clip:  clip 近场 ≥ north − 0.03 且 clip 在 r∈[32,64] 有覆盖(north 无) → 分级精度成立
"""
import argparse
import json

import numpy as np
import torch

from net.fovea_twotower.ego_map import EgoMapClip, EgoMapNaive, EgoMapNorth

C = 8
R_OBS = 10.0
BUCKETS = ((0, 4), (4, 8), (8, 16), (16, 32), (32, 64))


def run_episode(seed, T=300):
    rng = np.random.default_rng(seed)
    n_lm = 200
    lm_pos = rng.uniform(-55, 55, (n_lm, 2))
    lm_feat = rng.normal(size=(n_lm, C))
    lm_feat /= np.linalg.norm(lm_feat, axis=1, keepdims=True)

    maps = {"naive": EgoMapNaive(C, 64, 32.0),
            "north": EgoMapNorth(C, 64, 32.0),
            "clip": EgoMapClip(C, 32, 64.0, 3)}
    pos = np.zeros(2)
    yaw = 0.0
    seen = np.zeros(n_lm, bool)
    res = {n: {f"{lo}-{hi}": [] for lo, hi in BUCKETS} for n in maps}

    def evaluate():
        """只考看见过的地标;按当前距离分桶。"""
        rel = lm_pos - pos
        d = np.linalg.norm(rel, axis=1)
        cs, sn = np.cos(yaw), np.sin(yaw)
        for name, m in maps.items():
            if name == "naive":                      # 体坐标 = R(+yaw)·rel(校准见单测)
                q = np.stack([rel[:, 0] * cs - rel[:, 1] * sn,
                              rel[:, 0] * sn + rel[:, 1] * cs], -1)
            else:
                q = rel
            out = m.read(torch.from_numpy(q).float())
            sim = torch.nn.functional.cosine_similarity(
                out, torch.from_numpy(lm_feat).float(), dim=1).numpy()
            for lo, hi in BUCKETS:
                sel = (d >= lo) & (d < hi) & seen
                res[name][f"{lo}-{hi}"].extend(sim[sel].tolist())

    for t in range(T):
        dyaw = rng.uniform(-0.26, 0.26)
        yaw += dyaw
        speed = rng.uniform(0.5, 1.0)
        dpos = speed * np.array([np.sin(yaw), np.cos(yaw)])
        # 反弹墙:困在地标密集区
        for ax in (0, 1):
            if abs(pos[ax] + dpos[ax]) > 45:
                dpos[ax] = -dpos[ax]
                yaw = yaw + np.pi / 2
        pos += dpos
        cs, sn = np.cos(yaw), np.sin(yaw)
        dpos_body = np.array([dpos[0] * cs - dpos[1] * sn,
                              dpos[0] * sn + dpos[1] * cs])
        maps["naive"].step(dpos_body, dyaw)
        maps["north"].step(dpos)
        maps["clip"].step(dpos)
        rel = lm_pos - pos
        d = np.linalg.norm(rel, axis=1)
        vis = d < R_OBS
        if vis.any():
            seen |= vis
            feats = torch.from_numpy(lm_feat[vis]).float()
            rel_t = torch.from_numpy(rel[vis]).float()
            b = np.stack([rel[vis, 0] * cs - rel[vis, 1] * sn,
                          rel[vis, 0] * sn + rel[vis, 1] * cs], -1)
            maps["naive"].write(torch.from_numpy(b).float(), feats)
            maps["north"].write(rel_t, feats)
            maps["clip"].write(rel_t, feats)
        if t >= 100 and t % 25 == 0:
            evaluate()
    evaluate()
    return {n: {b: (float(np.mean(v)) if len(v) >= 3 else None)
                for b, v in bk.items()} for n, bk in res.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=12)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--out_json", default="runs/map_probe.json")
    args = p.parse_args()
    torch.manual_seed(0)

    acc = {}
    for s in range(args.seeds):
        r = run_episode(s, args.steps)
        for arm, bk in r.items():
            for b, v in bk.items():
                if v is not None:
                    acc.setdefault(arm, {}).setdefault(b, []).append(v)
    agg = {arm: {b: round(float(np.mean(v)), 3) for b, v in bk.items()}
           for arm, bk in acc.items()}

    near = lambda a: np.mean([x for b in ("0-4", "4-8")
                              for x in acc[a].get(b, [])])
    g_north = bool(near("north") >= near("naive") + 0.05)
    far_cov = agg["clip"].get("32-64")
    g_clip = bool(near("clip") >= near("north") - 0.03 and far_cov is not None
                  and far_cov > 0.3)
    out = dict(sim_by_dist=agg,
               near_field={"naive": round(float(near("naive")), 3),
                           "north": round(float(near("north")), 3),
                           "clip": round(float(near("clip")), 3)},
               gates={"G-north(north>=naive+0.05)": g_north,
                      "G-clip(clip>=north-0.03 且 32-64 有覆盖)": g_clip},
               budget={"naive": 4096, "north": 4096, "clip": 3072},
               seeds=args.seeds, steps=args.steps)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
