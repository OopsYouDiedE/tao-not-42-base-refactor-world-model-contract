# -*- coding: utf-8 -*-
"""Gate A 探针:冻结 Context 塔的历史状态里是否存有控制所需信息。

三个条件(前两个共用同一冻结塔,计算量/参数完全对等,只差历史):
  FULL     — 完整交错流(t-55..t 帧)末端 hidden(第 t 帧 81 token 均池);
  FRAME    — 只喂第 t 帧(零历史)同样均池;
  FRAME+ACT— FRAME 特征 ⊕ 过去 16 帧原始动作向量(384+384 维)。
    动作泄漏对照:流里含 efference copy,按键强自相关 ⇒ FULL 可纯靠转发
    过去动作赢 FRAME,不代表世界模型压缩了任何视觉历史。FULL 还得赢过
    "当前帧+裸动作历史"才证明状态里有超出动作缓冲的世界记忆。
线性 logistic 探针预测未来 0.5s(5 个区间):
  attack — 是否出现 key_attack;motion — 鼠标位移量是否高于训练集中位数。
判据(step1 §5,双层):
  verdict_history — auc_attack_FULL ≥ FRAME + 0.05(历史通道存在,原判据);
  verdict_world   — auc_attack_FULL ≥ FRAME+ACT + 0.02(世界记忆超出动作缓冲;
                    不过则播种收益可能仅是动作自相关,冻结世界塔的必要性存疑)。

用法:
    PYTHONPATH=. python train/fovea_twotower/probe_a.py \
        --data runs/data/g500_160p --ckpt runs/ftt_r1/ckpt.pt --n 2000
"""
import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from net.backbone import build_backbone
from net.config import BackboneConfig
from net.fovea_twotower import ContextTower
from train.fovea_twotower.data_utils import batch_to_stream
from train.gaming500.dataset import Gaming500Dataset, KEY_NAMES

I_ATTACK = KEY_NAMES.index("key_attack")
K_FUT = 5                                              # 未来区间数(0.5s@10Hz)
K_ACT = 16                                             # 裸动作历史帧数(1.6s,16×24=384 维)


@torch.no_grad()
def collect(model, dino, dl, n_max, dev):
    """→ feats_full [N,d], feats_frame [N,d], feats_act [N,K_ACT*24],
         feats_state [N,9*1536], y_attack [N], mouse_mag [N]

    feats_state = 9 层 GDN recurrent_state 沿 K 维均池([6,512,256]→[6,256])。
    这才是 R2 播种真正递交的通道;末层 hidden 均池(feats_full)只是它的有损代理
    ——90p 诊断显示 FULL 会稀释单帧信息,故闸门需同时探真实通道。"""
    F_full, F_frame, F_act, F_state, ya, mm = [], [], [], [], [], []
    for batch in dl:
        lat, act = batch_to_stream(batch, dino, dev)
        B, L = lat.shape[:2]
        t = L - 1 - K_FUT
        h_full, states = model.encode(lat[:, :t + 1], act[:, :t], want_states=True)
        F_full.append(h_full[:, -81:].mean(1).float().cpu())
        F_state.append(torch.cat(
            [st["recurrent_state"].mean(2).flatten(1) for st in states],
            1).float().cpu())
        h_frame, _ = model.encode(lat[:, t:t + 1], act[:, :0])
        F_frame.append(h_frame[:, -81:].mean(1).float().cpu())
        F_act.append(act[:, t - K_ACT:t].flatten(1).float().cpu())
        keys = batch["keys"][:, t:t + K_FUT]           # [B,K,20]
        ya.append(keys[..., I_ATTACK].any(1).float())
        mag = (batch["dx"].abs() + batch["dy"].abs())[:, t:t + K_FUT].mean(1)
        mm.append(mag)
        if sum(x.shape[0] for x in ya) >= n_max:
            break
    cat = lambda xs: torch.cat(xs).numpy()
    return cat(F_full), cat(F_frame), cat(F_act), cat(F_state), cat(ya), cat(mm)


def fit_auc(Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return float("nan")
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xtr), ytr)
    return roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="runs/data/g500_160p")
    p.add_argument("--ckpt", default="runs/ftt_r1/ckpt.pt")
    p.add_argument("--n", type=int, default=2000, help="每 split 样本数上限")
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--out", default=None, help="结果 JSON 路径(默认打印)")
    p.add_argument("--crop", default=None, choices=[None, "resize", "center", "random"],
                   help="默认沿用 ckpt 训练时的 crop 口径")
    args = p.parse_args()
    dev = "cuda"

    model = ContextTower().to(dev).bfloat16().eval()
    ck = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ck["model"])
    crop = args.crop or ck.get("args", {}).get("crop", "resize")
    dino = build_backbone(BackboneConfig(kind="dinov2"))[0].to(dev).eval()
    mk = lambda split: DataLoader(
        Gaming500Dataset(args.data, seq_len=args.seq, img_size=126,
                         stride=args.seq // 2, crop_mode=crop,
                         split=split, holdout_frac=0.1),
        batch_size=args.bs, shuffle=(split == "train"), num_workers=4)
    Xf_tr, Xr_tr, Xa_tr, Xs_tr, ya_tr, mm_tr = collect(model, dino, mk("train"), args.n, dev)
    Xf_te, Xr_te, Xa_te, Xs_te, ya_te, mm_te = collect(model, dino, mk("holdout"), args.n, dev)
    Xfa_tr = np.concatenate([Xr_tr, Xa_tr], 1)         # FRAME+ACT 对照
    Xfa_te = np.concatenate([Xr_te, Xa_te], 1)
    Xsfa_tr = np.concatenate([Xs_tr, Xr_tr, Xa_tr], 1)  # STATE⊕FRAME⊕ACT(边际价值形式)
    Xsfa_te = np.concatenate([Xs_te, Xr_te, Xa_te], 1)

    med = np.median(mm_tr)
    res = {
        "n_train": len(ya_tr), "n_test": len(ya_te),
        "attack_rate": round(float(ya_te.mean()), 4),
        "auc_attack_FULL": fit_auc(Xf_tr, ya_tr, Xf_te, ya_te),
        "auc_attack_FRAME": fit_auc(Xr_tr, ya_tr, Xr_te, ya_te),
        "auc_attack_FRAME_ACT": fit_auc(Xfa_tr, ya_tr, Xfa_te, ya_te),
        "auc_attack_STATE": fit_auc(Xs_tr, ya_tr, Xs_te, ya_te),
        "auc_attack_STATE_FRAME_ACT": fit_auc(Xsfa_tr, ya_tr, Xsfa_te, ya_te),
        "auc_motion_FULL": fit_auc(Xf_tr, mm_tr > med, Xf_te, mm_te > med),
        "auc_motion_FRAME": fit_auc(Xr_tr, mm_tr > med, Xr_te, mm_te > med),
        "auc_motion_FRAME_ACT": fit_auc(Xfa_tr, mm_tr > med, Xfa_te, mm_te > med),
    }
    res = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}
    d = res["auc_attack_FULL"] - res["auc_attack_FRAME"]
    dw = res["auc_attack_FULL"] - res["auc_attack_FRAME_ACT"]
    dws = res["auc_attack_STATE_FRAME_ACT"] - res["auc_attack_FRAME_ACT"]
    res["verdict_history"] = ("PASS" if d >= 0.05 else "FAIL") + f" (Δattack={d:+.4f})"
    res["verdict_world"] = ("PASS" if dw >= 0.02 else "FAIL") + f" (Δattack_vs_frame+act={dw:+.4f})"
    # 边际价值形式:探真实播种通道(9层recurrent_state),问它在帧+动作之上还有多少增量
    res["verdict_world_state"] = ("PASS" if dws >= 0.02 else "FAIL") + f" (Δ={dws:+.4f})"
    res["verdict"] = res["verdict_history"]            # 兼容原判据读法
    print("[GateA]", json.dumps(res, ensure_ascii=False), flush=True)
    if args.out:
        json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
