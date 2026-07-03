"""Gate 0 诊断:区分"信息不存在 / 对齐 bug / 线性读出不够"三种失败模式。

D1 幅度相关:corr(|dx|, ||Δz||_patch均值) 逐游戏——无相关⇒对齐/数据 bug;有⇒信息在。
D2 对齐检查:corr(|dx_j|, ||Δz_{j+off}||) 对 off∈{-2..2}——峰应在 off=0。
D3 特征互相关平移探针(免参数):z_t 与 z_{t+1} 的 patch 网格空间互相关 argmax
   → 估计整幅平移(patch 单位),corr(shift_x, dx)。翻译级运动若可见,此处应显著。
D4 MLP 探针(非线性上界):2 层 MLP 吃 Δz 描述子 → dx;对比 ridge。

用法: python tests/probe_g500_gate0_diag.py --feat-dir runs/g500_gates/feats
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_g500_common import load_segments, game_dx_std, descriptor  # noqa: E402


def xcorr_shift(f0, f1, max_shift=4):
    """patch 网格互相关平移估计。f0,f1 [B,C,G,G] → shift_x [B](patch 单位,亚 patch 抛物线插值)。"""
    B, C, G, _ = f0.shape
    f0n = (f0 - f0.mean(dim=(2, 3), keepdim=True))
    f1n = (f1 - f1.mean(dim=(2, 3), keepdim=True))
    scores = []
    shifts = list(range(-max_shift, max_shift + 1))
    for sx in shifts:                                  # 只测水平平移(dx)
        a = f0n[..., max(0, sx):G + min(0, sx)]
        b = f1n[..., max(0, -sx):G + min(0, -sx)]
        num = (a * b).sum(dim=(1, 2, 3))
        den = (a.pow(2).sum(dim=(1, 2, 3)).sqrt() *
               b.pow(2).sum(dim=(1, 2, 3)).sqrt()).clamp(min=1e-4)
        scores.append(num / den)
    S = torch.stack(scores, 1)                          # [B,n_shift]
    k = S.argmax(1)
    # 抛物线亚 patch 插值
    kl = (k - 1).clamp(min=0)
    kr = (k + 1).clamp(max=len(shifts) - 1)
    sl = S.gather(1, kl[:, None])[:, 0]
    sc = S.gather(1, k[:, None])[:, 0]
    sr = S.gather(1, kr[:, None])[:, 0]
    denom = (sl - 2 * sc + sr).clamp(max=-1e-4)
    frac = 0.5 * (sl - sr) / denom
    base = torch.tensor(shifts, dtype=torch.float32, device=k.device)[k]
    return base + frac.clamp(-0.5, 0.5)


def pearson(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", default="runs/g500_gates/feats")
    ap.add_argument("--out", default="runs/g500_gates/gate0_diag.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segs = load_segments(args.feat_dir)
    stds = game_dx_std(segs)
    rep = {}

    print("== D1 幅度相关 & D2 对齐峰 & D3 互相关平移 ==", flush=True)
    for s in segs:
        T = s["n"] - 1
        keep = s["gui"][:T] == 0
        f = torch.from_numpy(s["feats"].astype(np.float32))
        dz_norm = (f[1:T + 1] - f[:T]).flatten(1).norm(dim=1).numpy()[keep]
        adx = np.abs(s["dx"][:T])[keep]
        d1 = pearson(adx, dz_norm)
        # D2:错位相关(不过滤 gui,保持索引齐)
        dzn_all = (f[1:T + 1] - f[:T]).flatten(1).norm(dim=1).numpy()
        offs = {}
        for off in (-2, -1, 0, 1, 2):
            lo, hi = max(0, -off), min(T, T - off)
            offs[off] = round(pearson(np.abs(s["dx"][lo:hi]),
                                      dzn_all[lo + off:hi + off]), 3)
        # D3:互相关平移(分批算,GPU)
        sh = []
        for i0 in range(0, T, 512):
            a = f[i0:min(i0 + 512, T)].to(device)
            b = f[i0 + 1:min(i0 + 512, T) + 1].to(device)
            sh.append(xcorr_shift(a, b).cpu())
        sh = torch.cat(sh).numpy()[keep]
        d3 = pearson(s["dx"][:T][keep], sh)
        # 高运动子集上的 D3
        m = adx > np.percentile(adx[adx > 0], 50) if (adx > 0).sum() > 30 else adx > 0
        d3_hi = pearson(s["dx"][:T][keep][m], sh[m]) if m.sum() > 30 else float("nan")
        rep[s["seg"]] = dict(d1_mag=round(d1, 3), d2_offsets=offs,
                             d3_shift=round(d3, 3),
                             d3_shift_himotion=round(d3_hi, 3) if d3_hi == d3_hi else None)
        print(f"  {s['seg'][:40]:42s} D1={d1:.3f} D2={offs} D3={d3:.3f} "
              f"D3hi={d3_hi if d3_hi == d3_hi else float('nan'):.3f}", flush=True)

    # D4:MLP 探针(合池鼠标类游戏,terraria/ghost 除外)
    print("== D4 MLP 探针(非线性上界)==", flush=True)
    Xs, Ys, Sp = [], [], []
    for s in segs:
        if s["game"] in ("terraria", "ghost-of-tsushima"):
            continue
        T = s["n"] - 1
        keep = s["gui"][:T] == 0
        X = descriptor(s["feats"][:T][keep], s["feats"][1:T + 1][keep])
        Y = (s["dx"][:T][keep] / stds[s["game"]]).astype(np.float32)
        idx = np.arange(T)[keep]
        Xs.append(X)
        Ys.append(Y)
        Sp.append(idx + 1 < s["t_cut"])
    X = torch.from_numpy(np.concatenate(Xs))
    Y = torch.from_numpy(np.concatenate(Ys))[:, None]
    S = torch.from_numpy(np.concatenate(Sp))
    mu, sd = X[S].mean(0, True), X[S].std(0, True).clamp(min=1e-4)
    X = ((X - mu) / sd).to(device)
    Y = Y.to(device)
    S = S.to(device)
    torch.manual_seed(0)
    mlp = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 512), torch.nn.SiLU(),
        torch.nn.Linear(512, 256), torch.nn.SiLU(),
        torch.nn.Linear(256, 1)).to(device)
    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    ntr = int(S.sum())
    Xtr, Ytr = X[S], Y[S]
    for step in range(1500):
        i = torch.randint(0, ntr, (256,), device=device)
        loss = F.mse_loss(mlp(Xtr[i]), Ytr[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = mlp(X[~S])
        yte = Y[~S]
        r2 = float(1 - ((pred - yte) ** 2).sum() / ((yte - yte.mean()) ** 2).sum())
        mag = yte.abs().squeeze(1)
        thr = torch.quantile(mag[mag > 1e-6], 0.5) if (mag > 1e-6).sum() > 30 else 0
        m = mag > thr
        r2h = float(1 - ((pred[m] - yte[m]) ** 2).sum() /
                    ((yte[m] - yte[m].mean()) ** 2).sum()) if m.sum() > 30 else None
    rep["mlp_pooled_fps"] = {"r2_dx": round(r2, 3),
                             "r2_dx_p50": round(r2h, 3) if r2h is not None else None}
    print(f"  MLP(鼠标类合池) r2_dx={r2:.3f} r2_dx_p50={r2h}", flush=True)

    with open(args.out, "w") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
