"""Gate 0:Δz→dx 线性可解码性探针(2026-07-03 辩论收敛的开工前置闸门)。

问题:冻结 DINOv3 在 176px/16px patch 下,鼠标位移的微观效果在 token 空间里可见吗?
方法:ridge 闭式解,Δz 描述子(全局+行+列均值)→ dx/dy;holdout=每段时间尾部 15%。
对照:单帧 z_t 描述子(应显著更低)、目标 shuffle(应≈0)。
分层:全部转移 / |dx|>非零 p50 / |dx|>p90;逐游戏 + 合池。附 key_attack 线性 AUC。

判读口径:鼠标视角类游戏 R²(dx) 显著 >0(如 >0.3)且远高于单帧对照 → 通过;
接近 0 → 表征侧看不见动作效果,后续动力学/策略投入全部悬空(辩论 T4 风险)。

用法: python tests/probe_g500_gate0.py --feat-dir runs/g500_gates/feats
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_g500_common import (                         # noqa: E402
    load_segments, game_dx_std, descriptor, frame_descriptor)

LAMBDAS = (1e2, 1e3, 1e4, 1e5)


def ridge_fit_eval(Xtr, Ytr, Xte, Yte, device="cuda"):
    """多目标 ridge:train 内 85/15 选 λ,返回 holdout 上逐目标 R² 与预测。"""
    Xtr_t = torch.from_numpy(Xtr).to(device)
    Ytr_t = torch.from_numpy(Ytr).to(device)
    n_fit = int(Xtr.shape[0] * 0.85)
    Xf, Yf, Xv, Yv = Xtr_t[:n_fit], Ytr_t[:n_fit], Xtr_t[n_fit:], Ytr_t[n_fit:]
    mu, sd = Xf.mean(0, keepdim=True), Xf.std(0, keepdim=True).clamp(min=1e-4)
    Xf, Xv = (Xf - mu) / sd, (Xv - mu) / sd
    ym = Yf.mean(0, keepdim=True)
    G = Xf.T @ Xf                                      # [D,D] fp32
    b = Xf.T @ (Yf - ym)
    eye = torch.eye(G.shape[0], device=device)
    best = None
    for lam in LAMBDAS:
        W = torch.linalg.solve(G + lam * eye, b)
        pred_v = Xv @ W + ym
        sse = ((pred_v - Yv) ** 2).sum(0)
        sst = ((Yv - Yv.mean(0)) ** 2).sum(0).clamp(min=1e-4)
        r2 = float((1 - sse / sst).mean())
        if best is None or r2 > best[0]:
            best = (r2, W)
    W = best[1]
    Xte_t = (torch.from_numpy(Xte).to(device) - mu) / sd
    pred = (Xte_t @ W + ym).cpu().numpy()
    Yte_t = torch.from_numpy(Yte)
    sse = ((torch.from_numpy(pred) - Yte_t) ** 2).sum(0)
    sst = ((Yte_t - Yte_t.mean(0)) ** 2).sum(0).clamp(min=1e-4)
    return (1 - sse / sst).numpy(), pred


def auc(scores, labels):
    """线性打分的 ROC-AUC(秩统计)。"""
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels > 0.5
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def collect(segs, stds, use_frame_desc=False):
    """全部段 → (X, Y[dx,dy 归一], attack 键, |dx_n| 强度, game, split) 平表。gui 转移剔除。"""
    Xs, Ys, Ks, Gm, Sp = [], [], [], [], []
    for s in segs:
        T = s["n"] - 1
        keep = s["gui"][:T] == 0
        f0, f1 = s["feats"][:T][keep], s["feats"][1:T + 1][keep]
        X = frame_descriptor(f0) if use_frame_desc else descriptor(f0, f1)
        std = stds[s["game"]]
        Y = np.stack([s["dx"][:T][keep] / std, s["dy"][:T][keep] / std], 1)
        Xs.append(X.astype(np.float32))
        Ys.append(Y.astype(np.float32))
        Ks.append(s["keys"][:T][keep][:, 7].astype(np.float32))   # bit7=key_attack
        Gm.append(np.array([s["game"]] * keep.sum()))
        idx = np.arange(T)[keep]
        Sp.append(idx + 1 < s["t_cut"])                # True=train
    return (np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Ks),
            np.concatenate(Gm), np.concatenate(Sp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", default="runs/g500_gates/feats")
    ap.add_argument("--out", default="runs/g500_gates/gate0.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segs = load_segments(args.feat_dir)
    stds = game_dx_std(segs)
    print("逐游戏 dx 归一尺度:", {k: round(v, 1) for k, v in stds.items()}, flush=True)

    report = {}
    for tag, use_frame in (("dz", False), ("frame_ctrl", True)):
        X, Y, K, G, S = collect(segs, stds, use_frame_desc=use_frame)
        res = {}
        games = sorted(set(G))
        for scope in games + ["ALL"]:
            m = (G == scope) if scope != "ALL" else np.ones(len(G), bool)
            tr, te = m & S, m & ~S
            if tr.sum() < 200 or te.sum() < 50:
                continue
            r2, pred = ridge_fit_eval(X[tr], Y[tr], X[te], Y[te], device)
            entry = {"n_tr": int(tr.sum()), "n_te": int(te.sum()),
                     "r2_dx": round(float(r2[0]), 3), "r2_dy": round(float(r2[1]), 3)}
            # 运动分层(holdout 内,阈值取该 scope 非零 |dx| 分位)
            mag = np.abs(Y[te][:, 0])
            nz = mag[mag > 1e-6]
            if len(nz) > 30:
                for name, q in (("p50", 50), ("p90", 90)):
                    thr = np.percentile(nz, q)
                    sel = mag > thr
                    if sel.sum() >= 30:
                        yt, yp = Y[te][sel, 0], pred[sel, 0]
                        sse = ((yp - yt) ** 2).sum()
                        sst = max(((yt - yt.mean()) ** 2).sum(), 1e-4)
                        entry[f"r2_dx_{name}"] = round(float(1 - sse / sst), 3)
            res[scope] = entry
        report[tag] = res
        print(f"\n== {tag} ==")
        for k, v in res.items():
            print(f"  {k:24s} {v}")

    # shuffle 对照 + key_attack 线性 AUC(仅 Δz 描述子)
    X, Y, K, G, S = collect(segs, stds)
    rng = np.random.default_rng(0)
    Ysh = Y[rng.permutation(len(Y))]
    r2s, _ = ridge_fit_eval(X[S], Ysh[S], X[~S], Ysh[~S], device)
    report["shuffle_ctrl"] = {"r2_dx": round(float(r2s[0]), 3),
                              "r2_dy": round(float(r2s[1]), 3)}
    r2k, predk = ridge_fit_eval(X[S], K[S, None], X[~S], K[~S, None], device)
    report["key_attack"] = {"auc": round(auc(predk[:, 0], K[~S]), 3),
                            "pos_rate": round(float(K[~S].mean()), 3)}
    print("\nshuffle 对照:", report["shuffle_ctrl"])
    print("key_attack 线性 AUC:", report["key_attack"], flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
