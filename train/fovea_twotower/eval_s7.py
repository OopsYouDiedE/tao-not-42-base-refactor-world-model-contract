# -*- coding: utf-8 -*-
"""S7a(Step4 核心):预训练慢塔(W4/Nemotron-4B)是否解锁**带时**消息记忆。

与 probe_b.py(S4a)逐行同协议——同样本限制、同 bearing/age 标签、同拟合与判据——
唯一差异 = 被探塔:从零 58M ContextTower → 预训练 4B W4Adapter。
STATE = 21 层 Mamba2 最终 ssm_state,每层对非状态轴取均值 → (B, state_size),
        沿层拼接(池化配方在看结果前固定,见下 pool_ssm);
FRAME = 单帧(81 视觉 + 1 消息,无历史)last_hidden_state 均池;
MSG-HIST = 裸消息缓冲(信息上界,不经塔,同 probe_b)。
判据(step4 §2,先于结果登记):acc_STATE ≥ acc_FRAME+5pt 且 R²_age(STATE) CI>0。

用法(W4 训练完、GPU 空出后):
    PYTHONPATH=. python train/fovea_twotower/eval_s7.py \
        --ckpt runs/ftt_w4/ckpt.pt --out runs/ftt_s7_360.json --n 3000
"""
import argparse
import json
from importlib import import_module

import numpy as np
import torch
from torch.utils.data import DataLoader

from net.backbone import build_backbone
from net.config import BackboneConfig
from train.fovea_twotower.data_utils import batch_to_stream_msg
from train.fovea_twotower.model_utils import build_eval_model, pool_ssm
from train.fovea_twotower.train_w4 import MODEL_ID
from train.fovea_twotower.probe_b import fit_acc, fit_r2, K_BACK, K_FRESH
from train.gaming500.dataset import Gaming500Dataset, N_MSG


@torch.no_grad()
def collect(model, dino, dl, n_max, dev):
    """逐行复刻 probe_b.collect,STATE 换成 21 层 ssm_state 池化。"""
    F_state, F_frame, F_hist, yb, ya = [], [], [], [], []
    for batch in dl:
        lat, act, msg = batch_to_stream_msg(batch, dino, dev)
        B, L = lat.shape[:2]
        t = L - 1
        evt = (msg[..., :8].sum(-1) > 0).cpu().numpy()
        bear = msg[..., :8].argmax(-1).cpu().numpy()
        keep, lbl_b, lbl_a = [], [], []
        for i in range(B):
            if evt[i, t - K_FRESH + 1:t + 1].any():
                continue
            past = np.nonzero(evt[i, t - K_BACK:t - K_FRESH + 1])[0]
            if len(past) == 0:
                continue
            j = int(past[-1]) + (t - K_BACK)
            keep.append(i)
            lbl_b.append(int(bear[i, j]))
            lbl_a.append(t - j)
        if not keep:
            continue
        k = torch.tensor(keep, device=dev)
        _, states = model.encode(lat[k], act[k], msg[k], want_states=True)
        F_state.append(pool_ssm(states).cpu())
        h_fr, _ = model.encode(lat[k, t:t + 1], act[k, :0], msg[k, t:t + 1])
        F_frame.append(h_fr.float().mean(1).cpu())
        F_hist.append(msg[k, t - K_BACK:t + 1].flatten(1).float().cpu())
        yb += lbl_b
        ya += lbl_a
        if len(yb) >= n_max:
            break
    cat = lambda xs: torch.cat(xs).numpy()
    return (cat(F_state), cat(F_frame), cat(F_hist),
            np.array(yb), np.array(ya, np.float32))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="runs/data/g500_360p")
    p.add_argument("--ckpt", default="runs/ftt_w4/ckpt.pt")
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--out", default="runs/ftt_s7_360.json")
    args = p.parse_args()
    dev = "cuda"

    model, ck = build_eval_model(args.ckpt, dev)
    crop = ck.get("args", {}).get("crop", "center")
    dino = build_backbone(BackboneConfig(kind="dinov2"))[0].to(dev).eval()
    mk = lambda split: DataLoader(
        Gaming500Dataset(args.data, seq_len=args.seq, img_size=126,
                         stride=args.seq // 2, crop_mode=crop, periph=True,
                         split=split, holdout_frac=0.1),
        batch_size=args.bs, shuffle=(split == "train"), num_workers=8)
    Xs_tr, Xf_tr, Xh_tr, yb_tr, ya_tr = collect(model, dino, mk("train"), args.n, dev)
    Xs_te, Xf_te, Xh_te, yb_te, ya_te = collect(model, dino, mk("holdout"), args.n, dev)

    res = {"tower": "W4", "model_id": MODEL_ID, "ckpt_step": ck.get("step"),
           "train_args": ck.get("args", {}),
           "n_train": len(yb_tr), "n_test": len(yb_te),
           "chance_bearing": round(1.0 / 8, 4),
           "dim_STATE": int(Xs_tr.shape[1]), "dim_FRAME": int(Xf_tr.shape[1]),
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
    res["verdict_s7a"] = ("PASS" if ok else "FAIL") + \
        f" (Δbearing={d:+.4f}, r2_age_ci_lo={lo_s:+.4f})"
    print("[S7a]", json.dumps(res, ensure_ascii=False), flush=True)
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
