"""Gate 1+2:三臂动作条件小动力学(IG 开环对照)+ CAD 探针头(小规模判效)。

Gate 1(辩论收敛的抗作弊 IG 口径):同预算三臂——无动作/聚合动作/30Hz 子帧有序动作,
小因果 transformer 在 PCA-256 观测上预测 Δy;IG = EV(带动作) − EV(无动作),
teacher-forced + 开环 rollout EV@k 双口径。IG>0 且子帧臂≥聚合臂 ⇒ 动作条件有效。

Gate 2(CAD 探针可行性):
  C_act(Δy, a)  真配对 vs 负样本(easy=shuffle / hard=dx变号 / hard=子帧对调) → AUC
  C_real(y, Δy) 真转移 vs 模型预测;开环深度 k=1..8 的分数曲线应单调降,
                且与逐样本真误差相关(辩论 S2-2 验收口径的小规模版)
  D(y)          回归模型逐转移误差 → holdout Spearman

用法: python tests/probe_g500_gate12.py --feat-dir runs/g500_gates/feats
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_g500_common import (                         # noqa: E402
    ACTION_DIM, encode_action, fit_pca, game_dx_std, load_segments, pooled_frames)

SEQ = 16          # 每窗转移数(帧数 SEQ+1)
D_MODEL = 256
STEPS = 1200
BATCH = 64
CTX = 8           # 开环 rollout 的上下文帧数
MAX_K = 8


def build_windows(segs, stds):
    """全段 → (Y[N,257,256]... 不,逐窗索引)。返回 dict:y/act(none|agg|sub)/game/split。"""
    data = {"y": [], "none": [], "agg": [], "sub": [], "game": [], "train": []}
    for si, s in enumerate(segs):
        y = s["_y"]                                    # [n,256] 由 main 注入
        std = stds[s["game"]]
        T = s["n"] - 1
        for t0 in range(0, T - SEQ + 1, 8):
            span = slice(t0, t0 + SEQ)
            if s["gui"][span].any():                   # 窗口内含菜单转移则弃
                continue
            is_train = (t0 + SEQ) < s["t_cut"]
            is_hold = t0 >= s["t_cut"]
            if not (is_train or is_hold):
                continue                               # 跨切点的窗弃用
            data["y"].append(y[t0:t0 + SEQ + 1])
            for mode in ("none", "agg", "sub"):
                data[mode].append(np.stack(
                    [encode_action(s, j, std, mode) for j in range(t0, t0 + SEQ)]))
            data["game"].append(s["game"])
            data["train"].append(is_train)
    out = {k: np.stack(v) if k != "game" else np.array(v) for k, v in data.items()}
    out["train"] = out["train"].astype(bool)
    return out


class TinyDyn(nn.Module):
    """因果 transformer:token_t = proj(y_t) + act_mlp(a_t) + pos → 预测 Δy_t。"""

    def __init__(self, obs_dim, act_dim, d=D_MODEL, layers=4, heads=8):
        super().__init__()
        self.obs_in = nn.Linear(obs_dim, d)
        self.act_in = nn.Sequential(nn.Linear(act_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.zeros(1, SEQ, d))
        blk = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0,
                                         activation="gelu", batch_first=True,
                                         norm_first=True)
        self.enc = nn.TransformerEncoder(blk, layers)
        self.head = nn.Linear(d, obs_dim)
        mask = torch.triu(torch.full((SEQ, SEQ), float("-inf")), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, y_ctx, act):
        """y_ctx [B,SEQ,obs], act [B,SEQ,act] → Δy 预测 [B,SEQ,obs]。"""
        h = self.obs_in(y_ctx) + self.act_in(act) + self.pos
        h = self.enc(h, mask=self.mask)
        return self.head(h)


def ev(pred, target):
    """解释方差(persistence Δy=0 基线):1 − MSE/Var(target)。"""
    return float(1 - ((pred - target) ** 2).mean() /
                 (target ** 2).mean().clamp(min=1e-8))


def train_arm(data, mode, device, seed=0):
    """训一臂,返回 (model, 指标 dict)。"""
    torch.manual_seed(seed)
    Y = torch.from_numpy(data["y"]).float()
    A = torch.from_numpy(data[mode]).float()
    tr = torch.from_numpy(data["train"])
    model = TinyDyn(Y.shape[-1], A.shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    Ytr, Atr = Y[tr].to(device), A[tr].to(device)
    n = Ytr.shape[0]
    for step in range(STEPS):
        i = torch.randint(0, n, (BATCH,), device=device)
        y_w, a_w = Ytr[i], Atr[i]
        dy = y_w[:, 1:] - y_w[:, :-1]
        pred = model(y_w[:, :-1], a_w)
        loss = F.mse_loss(pred, dy)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    # teacher-forced holdout EV(整体+逐游戏)
    model.eval()
    Yte, Ate = Y[~tr].to(device), A[~tr].to(device)
    games_te = data["game"][~tr.numpy()]
    with torch.no_grad():
        dy = Yte[:, 1:] - Yte[:, :-1]
        pred = model(Yte[:, :-1], Ate)
        m = {"ev_tf": round(ev(pred, dy), 4)}
        for g in sorted(set(games_te)):
            sel = torch.from_numpy(games_te == g).to(device)
            m[f"ev_tf_{g}"] = round(ev(pred[sel], dy[sel]), 4)
        # 开环 rollout:前 CTX 帧真实,之后喂预测(动作始终给真)
        y_roll = Yte.clone()
        for t in range(CTX, SEQ):
            pred_t = model(y_roll[:, :SEQ], Ate)[:, t - 1]
            y_roll[:, t] = y_roll[:, t - 1] + pred_t
        for k in range(1, MAX_K + 1):
            t = CTX + k - 1
            if t > SEQ:
                break
            err = ((y_roll[:, t] - Yte[:, t]) ** 2).mean()
            base = ((Yte[:, CTX - 1] - Yte[:, t]) ** 2).mean().clamp(min=1e-8)
            m[f"ev_ol_k{k}"] = round(float(1 - err / base), 4)
    return model, m


def mlp(in_dim, out_dim=1):
    return nn.Sequential(nn.Linear(in_dim, 256), nn.SiLU(),
                         nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, out_dim))


def auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels > 0.5
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def gate2(data, model, device, rep):
    """CAD 探针:C_act / C_real / D。使用子帧臂模型的预测作 C_real 负样本。"""
    Y = torch.from_numpy(data["y"]).float()
    A = torch.from_numpy(data["sub"]).float()
    tr = torch.from_numpy(data["train"])
    B, S1, dy_dim = Y.shape[0], Y.shape[1], Y.shape[2]
    flat_y = Y[:, :-1].reshape(-1, dy_dim)
    flat_dy = (Y[:, 1:] - Y[:, :-1]).reshape(-1, dy_dim)
    flat_a = A.reshape(-1, A.shape[-1])
    flat_tr = tr[:, None].expand(-1, SEQ).reshape(-1)
    rng = np.random.default_rng(0)

    # ---- C_act:真 (Δy,a) vs 三类负样本 ----
    def negatives(a, kind):
        a = a.clone()
        if kind == "easy":                             # 全库 shuffle
            return a[torch.from_numpy(rng.permutation(len(a)))]
        if kind == "flip":                             # dx/dy 变号(槽位 0/1 + mulaw 2/3)
            for base in (0, 24, 48):                   # sub 布局每槽 24 维
                a[:, base:base + 4] = -a[:, base:base + 4]
            return a
        # swap:子帧 1↔2 对调(仅当两槽都有效;keys+位移全换)
        sw = a.clone()
        sw[:, 0:24], sw[:, 24:48] = a[:, 24:48], a[:, 0:24]
        return sw

    x_pos = torch.cat([flat_dy, flat_a], 1)
    torch.manual_seed(1)
    head = mlp(x_pos.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    xs_tr = x_pos[flat_tr].to(device)
    a_tr = flat_a[flat_tr]
    dy_tr = flat_dy[flat_tr]
    kinds = ("easy", "flip", "swap")
    for step in range(1200):
        i = torch.randint(0, xs_tr.shape[0], (256,))
        neg_a = negatives(a_tr[i], kinds[step % 3]).to(device)
        x = torch.cat([xs_tr[i.to(device)],
                       torch.cat([dy_tr[i].to(device), neg_a], 1)])
        lab = torch.cat([torch.ones(256), torch.zeros(256)]).to(device)
        loss = F.binary_cross_entropy_with_logits(head(x)[:, 0], lab)
        opt.zero_grad()
        loss.backward()
        opt.step()
    head.eval()
    rep["c_act"] = {}
    with torch.no_grad():
        dy_te, a_te = flat_dy[~flat_tr], flat_a[~flat_tr]
        s_pos = head(torch.cat([dy_te, a_te], 1).to(device))[:, 0].cpu().numpy()
        for kind in kinds:
            mask = np.ones(len(a_te), bool)
            if kind == "flip":                         # 只在有位移处有意义
                mask = (a_te[:, 0:2].abs().sum(1) + a_te[:, 24:26].abs().sum(1)
                        + a_te[:, 48:50].abs().sum(1)).numpy() > 1e-6
            if kind == "swap":                         # 两子槽有效且内容不同
                mask = ((a_te[:, 72] * a_te[:, 73]).numpy() > 0.5) & \
                       ((a_te[:, 0:24] - a_te[:, 24:48]).abs().sum(1).numpy() > 1e-6)
            if mask.sum() < 50:
                rep["c_act"][kind] = None
                continue
            s_neg = head(torch.cat([dy_te[mask], negatives(a_te[mask], kind)], 1
                                   ).to(device))[:, 0].cpu().numpy()
            rep["c_act"][kind] = {
                "auc": round(auc(np.concatenate([s_pos[mask], s_neg]),
                                 np.concatenate([np.ones(int(mask.sum())),
                                                 np.zeros(len(s_neg))])), 3),
                "n": int(mask.sum())}

    # ---- C_real:真 Δy vs 模型预测 Δy;开环深度分数曲线 ----
    model.eval()
    with torch.no_grad():
        pred_tr = model(Y[tr, :-1].to(device), A[tr].to(device)).cpu()
    fake_dy = pred_tr.reshape(-1, dy_dim)
    real_dy = (Y[tr, 1:] - Y[tr, :-1]).reshape(-1, dy_dim)
    ctx_y = Y[tr, :-1].reshape(-1, dy_dim)
    torch.manual_seed(2)
    creal = mlp(2 * dy_dim).to(device)
    opt = torch.optim.AdamW(creal.parameters(), lr=1e-3)
    for step in range(1200):
        i = torch.randint(0, ctx_y.shape[0], (256,))
        x = torch.cat([torch.cat([ctx_y[i], real_dy[i]], 1),
                       torch.cat([ctx_y[i], fake_dy[i]], 1)]).to(device)
        lab = torch.cat([torch.ones(256), torch.zeros(256)]).to(device)
        loss = F.binary_cross_entropy_with_logits(creal(x)[:, 0], lab)
        opt.zero_grad()
        loss.backward()
        opt.step()
    creal.eval()
    Yte, Ate = Y[~tr].to(device), A[~tr].to(device)
    with torch.no_grad():
        y_roll = Yte.clone()
        for t in range(CTX, SEQ):
            y_roll[:, t] = y_roll[:, t - 1] + model(y_roll[:, :SEQ], Ate)[:, t - 1]
        curve, corr_pts = {}, []
        s_real_ref = creal(torch.cat([Yte[:, CTX - 1],
                                      Yte[:, CTX] - Yte[:, CTX - 1]], 1))[:, 0]
        curve["real"] = round(float(torch.sigmoid(s_real_ref).mean()), 4)
        for k in range(1, MAX_K + 1):
            t = CTX + k - 1
            if t > SEQ - 1:
                break
            dy_k = y_roll[:, t] - y_roll[:, t - 1]
            s = creal(torch.cat([y_roll[:, t - 1], dy_k], 1))[:, 0]
            curve[f"k{k}"] = round(float(torch.sigmoid(s).mean()), 4)
            err = ((y_roll[:, t] - Yte[:, t]) ** 2).mean(1)
            corr_pts.append(torch.stack([torch.sigmoid(s), err], 1).cpu().numpy())
        pts = np.concatenate(corr_pts)
        rk_s = pts[:, 0].argsort().argsort()
        rk_e = pts[:, 1].argsort().argsort()
        rep["c_real"] = {"score_curve": curve,
                         "spearman_score_vs_err": round(float(np.corrcoef(rk_s, rk_e)[0, 1]), 3)}

    # ---- D 头:y_t → 模型误差回归 ----
    with torch.no_grad():
        pred_te = model(Y[~tr, :-1].to(device), Ate).cpu()
    err_te = ((pred_te - (Y[~tr, 1:] - Y[~tr, :-1])) ** 2).mean(-1).reshape(-1)
    ctx_te = Y[~tr, :-1].reshape(-1, dy_dim)
    n_fit = int(0.7 * len(err_te))
    torch.manual_seed(3)
    dhead = mlp(dy_dim).to(device)
    opt = torch.optim.AdamW(dhead.parameters(), lr=1e-3)
    tgt = err_te.log1p()
    for step in range(800):
        i = torch.randint(0, n_fit, (256,))
        loss = F.mse_loss(dhead(ctx_te[i].to(device))[:, 0], tgt[i].to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        p = dhead(ctx_te[n_fit:].to(device))[:, 0].cpu().numpy()
    rk_p, rk_t = p.argsort().argsort(), tgt[n_fit:].numpy().argsort().argsort()
    rep["d_head"] = {"spearman": round(float(np.corrcoef(rk_p, rk_t)[0, 1]), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", default="runs/g500_gates/feats")
    ap.add_argument("--out", default="runs/g500_gates/gate12.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segs = load_segments(args.feat_dir)
    stds = game_dx_std(segs)

    # 观测降维:4×4 池化 → PCA-256(只用 train 帧拟合)
    train_frames = np.concatenate(
        [pooled_frames(s["feats"][:s["t_cut"]]) for s in segs])
    mean, comps = fit_pca(train_frames, k=256)
    scale = None
    for s in segs:
        y = (pooled_frames(s["feats"]) - mean) @ comps
        if scale is None:
            scale = np.std((y[:len(y) * 85 // 100]), axis=0).mean() + 1e-6
        s["_y"] = (y / scale).astype(np.float32)
    data = build_windows(segs, stds)
    n_tr, n_te = int(data["train"].sum()), int((~data["train"]).sum())
    print(f"窗口:train={n_tr} holdout={n_te} (seq={SEQ})", flush=True)

    rep = {"windows": {"train": n_tr, "holdout": n_te}}
    models = {}
    for mode in ("none", "agg", "sub"):
        models[mode], m = train_arm(data, mode, device)
        rep[f"arm_{mode}"] = m
        print(f"[{mode}] {m}", flush=True)
    ig_tf = rep["arm_sub"]["ev_tf"] - rep["arm_none"]["ev_tf"]
    ig_ol = np.mean([rep["arm_sub"][f"ev_ol_k{k}"] - rep["arm_none"][f"ev_ol_k{k}"]
                     for k in (1, 2, 3, 4)])
    rep["IG"] = {"teacher_forced_sub": round(ig_tf, 4),
                 "teacher_forced_agg": round(rep["arm_agg"]["ev_tf"]
                                             - rep["arm_none"]["ev_tf"], 4),
                 "openloop_k1_4_sub": round(float(ig_ol), 4)}
    print("IG:", rep["IG"], flush=True)

    gate2(data, models["sub"], device, rep)
    print("C_act:", rep["c_act"], flush=True)
    print("C_real:", rep["c_real"], flush=True)
    print("D:", rep["d_head"], flush=True)
    with open(args.out, "w") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
