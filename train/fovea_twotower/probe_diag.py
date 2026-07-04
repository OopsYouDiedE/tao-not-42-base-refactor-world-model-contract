# -*- coding: utf-8 -*-
"""Gate A FAIL 归因诊断:分辨率盲 vs 线性读出弱 vs 状态真无信息。

在 probe_a 同一冻结塔/同一数据上补测:
  1. MLP 探针(非线性读出)— FRAME/FULL 预测 attack;
  2. FULL⊕ACT vs FRAME⊕ACT — 状态在裸动作之上的边际价值;
  3. onset 标签(过去3帧无开火 & 未来5帧有开火, 排除正在开火样本)
     — 把自相关从标签里洗掉后的真实判别力。
"""
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from net.fovea_twotower import ContextTower
from train.fovea_twotower.train_r1 import batch_to_stream
from train.gaming500.dataset import Gaming500Dataset, KEY_NAMES

I_ATTACK = KEY_NAMES.index("key_attack")
K_FUT, K_ACT, K_PAST = 5, 16, 3
DEV = "cuda"


@torch.no_grad()
def collect(model, dino, dl, n_max):
    Ff, Fr, Fa, ya, past = [], [], [], [], []
    for batch in dl:
        lat, act = batch_to_stream(batch, dino, DEV)
        B, L = lat.shape[:2]
        t = L - 1 - K_FUT
        h_full, _ = model.encode(lat[:, :t + 1], act[:, :t])
        Ff.append(h_full[:, -81:].mean(1).float().cpu())
        h_frame, _ = model.encode(lat[:, t:t + 1], act[:, :0])
        Fr.append(h_frame[:, -81:].mean(1).float().cpu())
        Fa.append(act[:, t - K_ACT:t].flatten(1).float().cpu())
        keys = batch["keys"]
        ya.append(keys[:, t:t + K_FUT, I_ATTACK].any(1).float())
        past.append(keys[:, t - K_PAST:t, I_ATTACK].any(1).float())
        if sum(x.shape[0] for x in ya) >= n_max:
            break
    cat = lambda xs: torch.cat(xs).numpy()
    return cat(Ff), cat(Fr), cat(Fa), cat(ya), cat(past)


def auc(Xtr, ytr, Xte, yte, kind="lin"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return float("nan")
    sc = StandardScaler().fit(Xtr)
    if kind == "lin":
        clf = LogisticRegression(max_iter=2000, C=1.0)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=300,
                            early_stopping=True, random_state=0)
    clf.fit(sc.transform(Xtr), ytr)
    return round(float(roc_auc_score(
        yte, clf.predict_proba(sc.transform(Xte))[:, 1])), 4)


def main():
    model = ContextTower().to(DEV).bfloat16().eval()
    model.load_state_dict(torch.load("runs/ftt_r1/ckpt.pt", map_location=DEV)["model"])
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                          verbose=False).to(DEV).eval()
    mk = lambda split: DataLoader(
        Gaming500Dataset("runs/data/g500_160p", seq_len=64, img_size=126,
                         stride=32, split=split, holdout_frac=0.1),
        batch_size=16, shuffle=(split == "train"), num_workers=4)
    Ff_tr, Fr_tr, Fa_tr, ya_tr, pa_tr = collect(model, dino, mk("train"), 2000)
    Ff_te, Fr_te, Fa_te, ya_te, pa_te = collect(model, dino, mk("holdout"), 2000)
    Ffa_tr = np.concatenate([Ff_tr, Fa_tr], 1)          # FULL⊕ACT
    Ffa_te = np.concatenate([Ff_te, Fa_te], 1)
    Fra_tr = np.concatenate([Fr_tr, Fa_tr], 1)          # FRAME⊕ACT
    Fra_te = np.concatenate([Fr_te, Fa_te], 1)

    res = {"n": [len(ya_tr), len(ya_te)]}
    # --- 1/2: attack 原标签 ---
    res["attack"] = {
        "FRAME_lin": auc(Fr_tr, ya_tr, Fr_te, ya_te),
        "FRAME_mlp": auc(Fr_tr, ya_tr, Fr_te, ya_te, "mlp"),
        "FULL_lin": auc(Ff_tr, ya_tr, Ff_te, ya_te),
        "FULL_mlp": auc(Ff_tr, ya_tr, Ff_te, ya_te, "mlp"),
        "ACT_lin": auc(Fa_tr, ya_tr, Fa_te, ya_te),
        "FRAME+ACT_lin": auc(Fra_tr, ya_tr, Fra_te, ya_te),
        "FULL+ACT_lin": auc(Ffa_tr, ya_tr, Ffa_te, ya_te),
    }
    # --- 3: onset 标签(排除正在开火样本) ---
    m_tr, m_te = pa_tr == 0, pa_te == 0
    res["onset_rate"] = [round(float(ya_tr[m_tr].mean()), 4),
                         round(float(ya_te[m_te].mean()), 4)]
    sel = lambda X, m: X[m]
    res["onset"] = {
        "FRAME_lin": auc(sel(Fr_tr, m_tr), ya_tr[m_tr], sel(Fr_te, m_te), ya_te[m_te]),
        "FRAME_mlp": auc(sel(Fr_tr, m_tr), ya_tr[m_tr], sel(Fr_te, m_te), ya_te[m_te], "mlp"),
        "FULL_lin": auc(sel(Ff_tr, m_tr), ya_tr[m_tr], sel(Ff_te, m_te), ya_te[m_te]),
        "FULL_mlp": auc(sel(Ff_tr, m_tr), ya_tr[m_tr], sel(Ff_te, m_te), ya_te[m_te], "mlp"),
        "ACT_lin": auc(sel(Fa_tr, m_tr), ya_tr[m_tr], sel(Fa_te, m_te), ya_te[m_te]),
        "FULL+ACT_lin": auc(sel(Ffa_tr, m_tr), ya_tr[m_tr], sel(Ffa_te, m_te), ya_te[m_te]),
        "FRAME+ACT_lin": auc(sel(Fra_tr, m_tr), ya_tr[m_tr], sel(Fra_te, m_te), ya_te[m_te]),
    }
    print("[DIAG]", json.dumps(res, ensure_ascii=False), flush=True)
    json.dump(res, open("runs/ftt_r1/gate_a_diag.json", "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
