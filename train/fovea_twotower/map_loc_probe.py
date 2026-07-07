#!/usr/bin/env python3
"""M3-loc 探针:①快塔用地图反向自定位(无真值)②慢塔查询 API 对 GT 准确率。

命题(用户 2026-07-08):快塔能否利用地图在没有位置真值的情况下推导自身位置?
机制:MC 里 yaw=本体感受精确,漂移只在平移(碰撞/动量)→ 北锚定下自定位
=纯平移扫描匹配:当前观测小图 vs 存图窗口互相关,峰=里程计误差,修正=虚拟 step。

里程计噪声模型(MC 现实):测量位移=真实×(1+ε),ε~N(0,0.1);碰撞 p=0.06
(真实位移清零、里程计以为走满——MC 撞墙的典型误差)。
两臂(唯一差异=是否启用 relocalize,每 5 步一次):
  dead   纯航位推算(对照,误差应无界增长)
  reloc  航位推算+地图互相关修正

门(先登记):
  G-loc1: reloc 末端定位误差中位 ≤ 0.4×dead
  G-loc2: reloc 全程误差有界:末端中位 ≤ 2.0 单位(dead 应远超)
  G-query: nearest() 距离误差中位 ≤ 2 格 且方位对(45°容差);survey() 未探索
           扇区查全率 ≥0.7(对 GT 访问覆盖)
附:回家向量=-p̂,其误差≡定位误差(自定位质量直接决定"记回家路线"质量)。
"""
import argparse
import json

import numpy as np
import torch

from net.fovea_twotower.ego_map import EgoMapNorthLoc, MapQuery, _bearing_cn

CLASSES = ["铁矿", "煤矿", "洞口"]
C = len(CLASSES)
R_OBS = 10.0


def selftest():
    """注入已知偏移,验证互相关恢复量与符号。"""
    rng = np.random.default_rng(0)
    m = EgoMapNorthLoc(C, 64, 32.0)
    pts = torch.from_numpy(rng.uniform(-20, 20, (30, 2))).float()
    f = torch.eye(C)[rng.integers(0, C, 30)].float()
    m.write(pts, f)
    # 模拟 e_now=-inj(观测相对存图偏移 -inj):正确修正量 ê=+inj(施加 p̂+=ê)
    inj = np.array([3.0, -2.0])
    est = m.relocalize(pts - torch.from_numpy(inj).float(), f)
    ok = est is not None and np.allclose(est, inj, atol=0.6)
    print(f"[selftest] e_now={-inj} 修正量 ê={est}(期望 {inj}) → "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def run_episode(seed, T=400, reloc=True, every=5, damp=1.0,
                ratio=1.15, subcell=False, min_pts=1.0, write_cap=0.0,
                size=64, window=4):
    rng = np.random.default_rng(seed)
    n_lm = 220
    lm_pos = rng.uniform(-55, 55, (n_lm, 2))
    lm_cls = rng.integers(0, C, n_lm)
    lm_feat = torch.eye(C)[lm_cls].float()

    m = EgoMapNorthLoc(C, size, 32.0, write_cap=write_cap)
    p_true = np.zeros(2)
    p_est = np.zeros(2)
    yaw = 0.0
    errs, seen = [], np.zeros(n_lm, bool)
    for t in range(T):
        yaw += rng.uniform(-0.26, 0.26)
        dpos = rng.uniform(0.5, 1.0) * np.array([np.sin(yaw), np.cos(yaw)])
        for ax in (0, 1):
            if abs(p_true[ax] + dpos[ax]) > 45:
                dpos[ax] = -dpos[ax]
                yaw += np.pi / 2
        collided = rng.random() < 0.06
        d_actual = np.zeros(2) if collided else dpos
        d_meas = dpos * (1 + rng.normal(0, 0.1, 2))    # 里程计:噪声+碰撞不知情
        p_true += d_actual
        p_est += d_meas
        m.step(d_meas)
        rel = lm_pos - p_true                           # 传感器看真实世界
        vis = np.linalg.norm(rel, axis=1) < R_OBS
        if vis.any():
            seen |= vis
            pts = torch.from_numpy(rel[vis]).float()
            fts = lm_feat[vis]
            if reloc and t % every == 0 and t > 10:
                e_hat = m.relocalize(pts, fts, window=window, min_ratio=ratio,
                                     subcell=subcell, min_pts=min_pts)
                if e_hat is not None:               # ê=修正量(坐标账见 mixin)
                    p_est += damp * e_hat
                    m.step(damp * e_hat)
            m.write(pts, fts)
        errs.append(float(np.linalg.norm(p_est - p_true)))

    # 查询 API 考核(对 GT):nearest 各类 + survey 覆盖
    q = MapQuery(m, CLASSES)
    qres = []
    rel = lm_pos - p_true
    d_all = np.linalg.norm(rel, axis=1)
    for k, cls in enumerate(CLASSES):
        cand = (lm_cls == k) & seen & (d_all < 30)
        if not cand.any():
            continue
        gt_i = np.argmin(np.where(cand, d_all, np.inf))
        v, _txt = q.nearest(cls)
        if v is None:
            qres.append(dict(cls=cls, found=False))
            continue
        derr = abs(float(np.linalg.norm(v)) - float(d_all[gt_i]))
        berr = _bearing_cn(v) == _bearing_cn(rel[gt_i])
        qres.append(dict(cls=cls, found=True, dist_err=round(derr, 2),
                         bearing_ok=bool(berr)))
    # survey 查全率:GT 未访问扇区(按 seen 地标覆盖)有多少被报出
    ang = (np.degrees(np.arctan2(rel[:, 0], rel[:, 1])) + 360) % 360
    sec = ((ang + 22.5) // 45).astype(int) % 8
    names8 = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]
    in32 = d_all < 30
    gt_unexp = {names8[s] for s in range(8)
                if seen[(sec == s) & in32].sum() <= 1}
    rep_unexp, _ = q.survey()
    recall = (len(gt_unexp & set(rep_unexp)) / len(gt_unexp)
              if gt_unexp else None)
    return errs, qres, recall


CONFIGS = {                       # 标定网格(种子0-9),最优上留出种子确认
    "e5_raw": dict(every=5, damp=1.0, ratio=1.15, subcell=False),
    "e5_sub": dict(every=5, damp=1.0, ratio=1.15, subcell=True),
    "e5_damp": dict(every=5, damp=0.6, ratio=1.25, subcell=True, min_pts=3.0),
    "e3_damp": dict(every=3, damp=0.5, ratio=1.25, subcell=True, min_pts=3.0),
    "e10_raw": dict(every=10, damp=1.0, ratio=1.15, subcell=False),
    "e5_anchor": dict(every=5, damp=1.0, ratio=1.15, subcell=False,
                      write_cap=6.0),
    "e3_anchor": dict(every=3, damp=1.0, ratio=1.15, subcell=False,
                      write_cap=6.0),
    "e5_fine": dict(every=5, damp=1.0, ratio=1.15, subcell=False,
                    size=128, window=6),
    "e3_fine": dict(every=3, damp=1.0, ratio=1.15, subcell=False,
                    size=128, window=6),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--mode", choices=["calib", "confirm"], default="confirm")
    p.add_argument("--config", default="e5_raw")
    p.add_argument("--seed0", type=int, default=100,
                   help="confirm 模式的留出种子起点")
    p.add_argument("--out_json", default="runs/map_loc_probe.json")
    args = p.parse_args()
    torch.manual_seed(0)
    st = selftest()

    if args.mode == "calib":
        table = {}
        for name, cfg in CONFIGS.items():
            fe = [run_episode(s, args.steps, reloc=True, **cfg)[0][-1]
                  for s in range(args.seeds)]
            table[name] = round(float(np.median(fe)), 2)
        dead = [run_episode(s, args.steps, reloc=False)[0][-1]
                for s in range(args.seeds)]
        table["dead"] = round(float(np.median(dead)), 2)
        print(json.dumps(dict(calib=table), ensure_ascii=False), flush=True)
        return

    cfg = CONFIGS[args.config]
    final = {"dead": [], "reloc": []}
    curves = {"dead": [], "reloc": []}
    all_q, all_rec = [], []
    for s in range(args.seed0, args.seed0 + args.seeds):
        for arm, rl in (("dead", False), ("reloc", True)):
            kw = cfg if rl else {}
            errs, qres, rec = run_episode(s, args.steps, reloc=rl, **kw)
            final[arm].append(errs[-1])
            curves[arm].append([errs[i] for i in (99, 199, 299, args.steps - 1)])
            if rl:
                all_q.extend(qres)
                if rec is not None:
                    all_rec.append(rec)
    med = {a: round(float(np.median(v)), 2) for a, v in final.items()}
    cur = {a: [round(float(x), 2) for x in np.median(np.array(v), 0)]
           for a, v in curves.items()}
    found = [x for x in all_q if x.get("found")]
    derr_med = (round(float(np.median([x["dist_err"] for x in found])), 2)
                if found else None)
    b_ok = (round(float(np.mean([x["bearing_ok"] for x in found])), 2)
            if found else None)
    rec_m = round(float(np.mean(all_rec)), 2) if all_rec else None
    gates = {
        "G-selftest": bool(st),
        "G-loc1(reloc<=0.4×dead)": bool(med["reloc"] <= 0.4 * med["dead"]),
        "G-loc2(reloc末端<=2.0)": bool(med["reloc"] <= 2.0),
        "G-query(nearest距离误差<=2且方位>=0.8;survey查全>=0.7)":
            bool(derr_med is not None and derr_med <= 2.0
                 and (b_ok or 0) >= 0.8 and (rec_m or 0) >= 0.7),
    }
    out = dict(final_err_med=med, err_curve_100_200_300_end=cur,
               query=dict(nearest_dist_err_med=derr_med,
                          bearing_acc=b_ok, n_found=len(found),
                          survey_recall=rec_m),
               gates=gates, config=args.config, holdout_seed0=args.seed0,
               seeds=args.seeds, steps=args.steps)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
