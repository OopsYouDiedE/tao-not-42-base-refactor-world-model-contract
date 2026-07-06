# -*- coding: utf-8 -*-
"""Probe B(S4a):冻结 W1 塔的状态里是否有**带时间**的消息记忆。

样本限制(强制走记忆,当前帧/当前消息均无事件可抄):
    最近周边事件落在 [t-30, t-2](0.2~3s 前),且 [t-1, t] 无事件。
特征三条件:
    STATE    — 9 层 recurrent_state 沿 K 均池(播种真实通道,step1 教训);
    FRAME    — 单帧(含当前消息 token,按限制其中无事件)零历史同型编码;
    MSG-HIST — 裸消息缓冲 [t-30..t] 展平(信息上界对照,不经塔)。
标签:
    bearing — 最近事件的 8 方位(多分类 acc,机会率 1/8);
    age     — 事件距今帧数,Ridge 回归 log1p(age) 的 R²(带时记忆的直接判据)。
判据(step2 §4):acc_STATE ≥ acc_FRAME+5pt 且 R²_age(STATE) bootstrap CI>0。

用法:
    PYTHONPATH=. python train/fovea_twotower/probe_b.py \
        --data runs/data/g500_360p --ckpt runs/ftt_w1/ckpt.pt --n 3000
"""
import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from net.backbone import build_backbone
from net.config import BackboneConfig
from net.fovea_twotower import ContextTower
from train.fovea_twotower.data_utils import batch_to_stream_msg
from train.gaming500.dataset import Gaming500Dataset, N_MSG

K_BACK = 30                                            # 事件回溯窗(3s)
K_FRESH = 2                                            # 近端禁区(强制记忆)


@torch.no_grad()
def collect(model, dino, dl, n_max, dev):
    F_state, F_frame, F_hist, yb, ya = [], [], [], [], []
    for batch in dl:
        lat, act, msg = batch_to_stream_msg(batch, dino, dev)
        B, L = lat.shape[:2]
        t = L - 1
        evt = (msg[..., :8].sum(-1) > 0).cpu().numpy() # [B,L]
        bear = msg[..., :8].argmax(-1).cpu().numpy()
        keep, lbl_b, lbl_a = [], [], []
        for i in range(B):
            if evt[i, t - K_FRESH + 1:t + 1].any():
                continue                               # 近端有事件 → 可抄,弃
            past = np.nonzero(evt[i, t - K_BACK:t - K_FRESH + 1])[0]
            if len(past) == 0:
                continue                               # 回溯窗无事件,弃
            j = int(past[-1]) + (t - K_BACK)           # 最近事件帧
            keep.append(i)
            lbl_b.append(int(bear[i, j]))
            lbl_a.append(t - j)
        if not keep:
            continue
        k = torch.tensor(keep, device=dev)
        _, states = model.encode(lat[k], act[k], msg[k], want_states=True)
        F_state.append(torch.cat(
            [st["recurrent_state"].mean(2).flatten(1) for st in states],
            1).float().cpu())
        h_fr, _ = model.encode(lat[k, t:t + 1], act[k, :0], msg[k, t:t + 1])
        F_frame.append(h_fr.mean(1).float().cpu())
        F_hist.append(msg[k, t - K_BACK:t + 1].flatten(1).float().cpu())
        yb += lbl_b
        ya += lbl_a
        if len(yb) >= n_max:
            break
    cat = lambda xs: torch.cat(xs).numpy()
    return (cat(F_state), cat(F_frame), cat(F_hist),
            np.array(yb), np.array(ya, np.float32))


def fit_acc(Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xtr), ytr)
    return float((clf.predict(sc.transform(Xte)) == yte).mean())


def fit_r2(Xtr, ytr, Xte, yte, boot=300):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    rg = Ridge(alpha=10.0).fit(sc.transform(Xtr), np.log1p(ytr))
    pred = rg.predict(sc.transform(Xte))
    yt = np.log1p(yte)
    r2 = lambda p, y: 1 - ((p - y) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9)
    rng = np.random.default_rng(0)
    bs = [r2(pred[i], yt[i]) for i in
          (rng.integers(0, len(yt), len(yt)) for _ in range(boot))]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(r2(pred, yt)), float(lo), float(hi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="runs/data/g500_360p")
    p.add_argument("--ckpt", default="runs/ftt_w1/ckpt.pt")
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    dev = "cuda"

    ck = torch.load(args.ckpt, map_location=dev)
    model = ContextTower(n_msg=N_MSG,
                         aux_msg=ck.get("args", {}).get("aux_msg", 0.0)
                         ).to(dev).bfloat16().eval()
    model.load_state_dict(ck["model"])
    crop = ck.get("args", {}).get("crop", "center")
    dino = build_backbone(BackboneConfig(kind="dinov2"))[0].to(dev).eval()
    mk = lambda split: DataLoader(
        Gaming500Dataset(args.data, seq_len=args.seq, img_size=126,
                         stride=args.seq // 2, crop_mode=crop, periph=True,
                         split=split, holdout_frac=0.1),
        batch_size=args.bs, shuffle=(split == "train"), num_workers=8)
    Xs_tr, Xf_tr, Xh_tr, yb_tr, ya_tr = collect(model, dino, mk("train"), args.n, dev)
    Xs_te, Xf_te, Xh_te, yb_te, ya_te = collect(model, dino, mk("holdout"), args.n, dev)

    res = {"n_train": len(yb_tr), "n_test": len(yb_te),
           "chance_bearing": round(1.0 / 8, 4),
           "acc_bearing_STATE": round(fit_acc(Xs_tr, yb_tr, Xs_te, yb_te), 4),
           "acc_bearing_FRAME": round(fit_acc(Xf_tr, yb_tr, Xf_te, yb_te), 4),
           "acc_bearing_MSGHIST": round(fit_acc(Xh_tr, yb_tr, Xh_te, yb_te), 4)}
    r2s, lo_s, hi_s = fit_r2(Xs_tr, ya_tr, Xs_te, ya_te)
    r2f, lo_f, hi_f = fit_r2(Xf_tr, ya_tr, Xf_te, ya_te)
    r2h, lo_h, hi_h = fit_r2(Xh_tr, ya_tr, Xh_te, ya_te)
    res.update(r2_age_STATE=round(r2s, 4), r2_age_STATE_ci=[round(lo_s, 4), round(hi_s, 4)],
               r2_age_FRAME=round(r2f, 4), r2_age_FRAME_ci=[round(lo_f, 4), round(hi_f, 4)],
               r2_age_MSGHIST=round(r2h, 4), r2_age_MSGHIST_ci=[round(lo_h, 4), round(hi_h, 4)])
    d = res["acc_bearing_STATE"] - res["acc_bearing_FRAME"]
    ok = d >= 0.05 and lo_s > 0
    res["verdict_s4a"] = ("PASS" if ok else "FAIL") + \
        f" (Δbearing={d:+.4f}, r2_age_ci_lo={lo_s:+.4f})"
    print("[ProbeB]", json.dumps(res, ensure_ascii=False), flush=True)
    if args.out:
        json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
