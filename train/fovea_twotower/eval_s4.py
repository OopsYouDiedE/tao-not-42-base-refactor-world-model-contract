# -*- coding: utf-8 -*-
"""S4b/S4c 终判:M1(真实消息) vs M0(置零) vs Mscr(时序打乱)。

同一 holdout 顺序 + 同一采样噪声,逐窗配对。主指标(step2 §4 预登记):
    dx 转向方向 AUC,限"周边事件后 1s 内且有明确转向"的窗口;
    S4b = Δ(M1−M0) bootstrap CI>0;S4c = Δ(M1−Mscr) CI>0。
特异性检查:同指标在**无事件窗**上的 Δ 应明显小于事件窗(否则收益与消息无关)。
次指标:事件窗 attack AUC;全体 keys 宏 AUC(报告)。
"""
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from net.fovea_twotower import ActionTower, ContextTower
from train.fovea_twotower.train_w2 import H, prep
from train.gaming500.dataset import Gaming500Dataset, KEY_NAMES, N_MSG

I_ATT = 2 + KEY_NAMES.index("key_attack")
DEV = "cuda"
DATA = "runs/data/g500_360p"
CTX = "runs/ftt_w1/ckpt.pt"
K_EVT = 10                                             # 事件回溯窗(1s)
DX_TH = 0.5                                            # 明确转向阈(featurize 域)


@torch.no_grad()
def gather(model, ctx, dino, dl, mode, n_max=2000):
    gen = torch.Generator(DEV).manual_seed(1234)
    sgen = torch.Generator().manual_seed(777)
    DX, DXp, ATT, ATTp, KP, KT, EV = [], [], [], [], [], [], []
    n = 0
    for batch in dl:
        lat_now, states, z1, msg_hist = prep(batch, dino, ctx, mode, sgen, DEV)
        pred = model.sample(lat_now, seed=states, steps=4, generator=gen)
        DX.append(z1[..., 0].sum(1).float().cpu())     # 目标 dx 和(featurize 域)
        DXp.append(pred[..., 0].sum(1).float().cpu())
        ATT.append((z1[..., I_ATT] > 0.5).float().cpu())
        ATTp.append(pred[..., I_ATT].float().cpu())
        KP.append(pred[..., 2:22].float().cpu())
        KT.append((z1[..., 2:22] > 0.5).float().cpu())
        EV.append((msg_hist[:, -K_EVT:, :8].sum((1, 2)) > 0).float().cpu())
        n += z1.shape[0]
        if n >= n_max:
            break
    c = lambda xs: torch.cat(xs).numpy()
    return c(DX), c(DXp), c(ATT), c(ATTp), c(KP), c(KT), c(EV)


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    y, s = np.asarray(y).ravel(), np.asarray(s).ravel()
    if len(np.unique(y)) < 2:
        return float("nan")
    return roc_auc_score(y, s)


def keys_auc(TK, PK, idx=None):
    if idx is not None:
        TK, PK = TK[idx], PK[idx]
    vals = [auc(TK[..., k], PK[..., k]) for k in range(TK.shape[-1])]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def load_tower(path):
    m = ActionTower(horizon=H).to(DEV).bfloat16()
    m.load_state_dict(torch.load(path, map_location=DEV)["model"])
    return m.eval()


def paired_ci(fn_a, fn_b, N, boot=500):
    rng = np.random.default_rng(0)
    ds = []
    for _ in range(boot):
        i = rng.integers(0, N, N)
        d = fn_a(i) - fn_b(i)
        if np.isfinite(d):
            ds.append(d)
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    torch.manual_seed(0)
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                          verbose=False).to(DEV).eval()
    ck = torch.load(CTX, map_location=DEV)
    ctx = ContextTower(n_msg=N_MSG).to(DEV).bfloat16().eval()
    ctx.load_state_dict(ck["model"])
    crop = ck.get("args", {}).get("crop", "center")
    mk = lambda: DataLoader(
        Gaming500Dataset(DATA, seq_len=64, img_size=126, stride=32,
                         crop_mode=crop, periph=True, split="holdout",
                         holdout_frac=0.1),
        batch_size=8, shuffle=False, num_workers=8)
    arms = {}
    for name, path, mode in [("M1", "runs/ftt_w2_m1/ckpt.pt", 1),
                             ("M0", "runs/ftt_w2_m0/ckpt.pt", 0),
                             ("Mscr", "runs/ftt_w2_mscr/ckpt.pt", 2)]:
        arms[name] = gather(load_tower(path), ctx, dino, mk(), mode)
    dx1 = arms["M1"][0]
    assert all(np.allclose(dx1, arms[a][0]) for a in arms), "窗口未对齐"
    ev = arms["M1"][6] > 0
    turn = np.abs(dx1) > DX_TH                         # 有明确转向
    lbl_dir = (dx1 > 0).astype(float)

    res = {"n_windows": int(len(dx1)), "n_event": int(ev.sum()),
           "n_event_turn": int((ev & turn).sum())}

    # 逐窗配对 bootstrap:在子集内重采样
    for tag, m in [("evt", ev & turn), ("noevt", (~ev) & turn)]:
        sub = np.nonzero(m)[0]
        y = lbl_dir[sub]
        Ns = len(sub)
        res[f"auc_dxdir_{tag}"] = {a: round(auc(y, arms[a][1][sub]), 4) for a in arms}
        for a, b, key in [("M1", "M0", "s4b"), ("M1", "Mscr", "s4c")]:
            sa, sb = arms[a][1][sub], arms[b][1][sub]
            lo, hi = paired_ci(lambda i: auc(y[i], sa[i]),
                               lambda i: auc(y[i], sb[i]), Ns)
            d = auc(y, sa) - auc(y, sb)
            res[f"delta_{key}_{tag}"] = [round(d, 4), round(lo, 4), round(hi, 4)]
    # 次指标:事件窗 attack;全体 keys
    sub = np.nonzero(ev)[0]
    res["auc_attack_evt"] = {a: round(auc(arms[a][2][sub], arms[a][3][sub]), 4)
                             for a in arms}
    res["auc_keys_all"] = {a: round(keys_auc(arms[a][5], arms[a][4]), 4) for a in arms}
    res["verdict_s4b"] = ("PASS" if res["delta_s4b_evt"][1] > 0 else "FAIL") + \
        f" (Δ={res['delta_s4b_evt'][0]:+.4f}, CI=[{res['delta_s4b_evt'][1]:+.4f},{res['delta_s4b_evt'][2]:+.4f}])"
    res["verdict_s4c"] = ("PASS" if res["delta_s4c_evt"][1] > 0 else "FAIL") + \
        f" (Δ={res['delta_s4c_evt'][0]:+.4f}, CI=[{res['delta_s4c_evt'][1]:+.4f},{res['delta_s4c_evt'][2]:+.4f}])"
    print("[S4]", json.dumps(res, ensure_ascii=False), flush=True)
    json.dump(res, open("runs/ftt_s4_360.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
