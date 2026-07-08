# -*- coding: utf-8 -*-
"""S7b(Step4):预训练慢塔(W4/Nemotron-4B)是否**扩张**脑内世界(记忆视界+可控性)。

与 eval_s5.py(S5,W1 从零塔)逐行同协议——同锚点 T_ANCHOR、同 k、同 ridge/CI、同 S5c——
唯一差异 = 被探塔:从零 58M ContextTower → 预训练 4B W4Adapter,且 STATE 换成 21 层
Mamba2 ssm_state 池化(pool_ssm,与 eval_s7 同配方)。

判据(step4 §2,先于结果登记):
  S7b-horizon:k=10 记忆视界 ΔR²(W4 的 STATE−FRAME 差)> W1 的同量(读 runs/ftt_s5_360.json)。
  S7b-control:S5c 可控性 Δerr(flip−true) CI 下界 > 0(W4 自身)。
  两半皆过 = S7b PASS(脑内世界较从零塔扩张)。

用法(W4/W4b 训练完、GPU 空出后):
    PYTHONPATH=. python train/fovea_twotower/eval_s7b.py \
        --ckpt runs/ftt_w4b/ckpt.pt --w1 runs/ftt_s5_360.json \
        --out runs/ftt_s7b_360.json
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from net.backbone import build_backbone
from net.config import BackboneConfig
from train.fovea_twotower.data_utils import batch_to_stream_msg
from train.fovea_twotower.eval_utils import pool9, ridge_r2, paired_r2_ci
from train.fovea_twotower.model_utils import build_eval_model, pool_ssm
from train.fovea_twotower.train_w4 import MODEL_ID
from train.gaming500.dataset import Gaming500Dataset

DEV = "cuda"
DATA = "runs/data/g500_360p"
KS = [2, 5, 10, 20]                                    # 帧;10Hz → 0.2/0.5/1/2s
T_ANCHOR = 43                                          # 64 帧窗:前 44 帧前缀,留 20 帧未来
DX_TH = 0.5                                            # 可控性锚点阈(featurize 域,同 eval_s5)
P = 81 + 1 + 1                                         # 帧块周期(vis|msg|act)= 83


@torch.no_grad()
def collect(model, dino, dl, n_max, dev):
    """逐行复刻 eval_s5.collect,STATE 换成 pool_ssm(21 层 ssm_state)。"""
    t = T_ANCHOR
    S, FR, PLAN, Y = [], [], {k: [] for k in KS}, {k: {} for k in KS}
    E_true, E_flip, DXA = [], [], []
    n = 0
    for batch in dl:
        lat, act, msg = batch_to_stream_msg(batch, dino, dev)
        B = lat.shape[0]
        # ── 特征臂 ──
        _, states = model.encode(lat[:, :t + 1], act[:, :t], msg[:, :t + 1],
                                 want_states=True)
        S.append(pool_ssm(states).float().cpu())
        h_fr, _ = model.encode(lat[:, t:t + 1], act[:, :0], msg[:, t:t + 1])
        FR.append(h_fr.float().mean(1).cpu())
        for k in KS:
            PLAN[k].append(act[:, t:t + k].flatten(1).float().cpu())
        # ── 目标(过 tgt_norm,与训练目标同域)──
        zn = model.tgt_norm(lat)
        for k in KS:
            Y[k].setdefault("past", []).append(pool9(zn[:, t - k]).float().cpu())
            Y[k].setdefault("fut", []).append(pool9(zn[:, t + k]).float().cpu())
        Y[KS[0]].setdefault("now", []).append(pool9(zn[:, t]).float().cpu())
        # ── S5c 可控性:末动作真实 vs dx 翻符号,预测下帧首 patch ──
        lat_cf = torch.cat([lat[:, :t + 1], torch.zeros_like(lat[:, :1])], 1)
        msg_cf = torch.cat([msg[:, :t + 1], torch.zeros_like(msg[:, :1])], 1)
        tgt0 = zn[:, t + 1, 0]                          # [B,384]
        pos = P * t + 81 + 1                            # act_t 位(预测 t+1 vis0)
        errs = []
        for flip in (False, True):
            a = act[:, :t + 1].clone()
            if flip:
                a[:, t, 0] = -a[:, t, 0]
            h, _ = model.encode(lat_cf, a, msg_cf)
            pred = model.head(h[:, pos])
            errs.append(((pred.float() - tgt0.float()) ** 2).mean(-1).cpu())
        E_true.append(errs[0])
        E_flip.append(errs[1])
        DXA.append(act[:, t, 0].abs().float().cpu())
        n += B
        if n >= n_max:
            break
    c = lambda xs: torch.cat(xs).numpy()
    out = {"S": c(S), "FR": c(FR),
           "PLAN": {k: c(v) for k, v in PLAN.items()},
           "e_true": c(E_true), "e_flip": c(E_flip), "dxa": c(DXA),
           "y_now": c(Y[KS[0]]["now"])}
    for k in KS:
        out[f"y_past{k}"] = c(Y[k]["past"])
        out[f"y_fut{k}"] = c(Y[k]["fut"])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/ftt_w4b/ckpt.pt")
    p.add_argument("--w1", default="runs/ftt_s5_360.json",
                   help="W1 从零塔 S5 结果(做 S7b-horizon 的跨模型对照)")
    p.add_argument("--out", default="runs/ftt_s7b_360.json")
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-test", type=int, default=1200)
    p.add_argument("--bs", type=int, default=4)
    args = p.parse_args()
    torch.manual_seed(0)

    model, ck = build_eval_model(args.ckpt, DEV)
    crop = ck.get("args", {}).get("crop", "center")
    dino = build_backbone(BackboneConfig(kind="dinov2"))[0].to(DEV).eval()
    mk = lambda split, sh: DataLoader(
        Gaming500Dataset(DATA, seq_len=64, img_size=126, stride=32,
                         crop_mode=crop, periph=True, split=split,
                         holdout_frac=0.1),
        batch_size=args.bs, shuffle=sh, num_workers=8)
    tr = collect(model, dino, mk("train", True), args.n_train, DEV)
    te = collect(model, dino, mk("holdout", False), args.n_test, DEV)
    print(f"[S7b] n_train={len(tr['dxa'])} n_test={len(te['dxa'])}", flush=True)

    res = {"tower": "W4", "model_id": MODEL_ID, "ckpt_step": ck.get("step"),
           "train_args": ck.get("args", {}),
           "n_train": len(tr["dxa"]), "n_test": len(te["dxa"]),
           "dim_STATE": int(tr["S"].shape[1]), "dim_FRAME": int(tr["FR"].shape[1])}
    for k in KS:
        for side in ("past", "fut"):
            ytr, yte = tr[f"y_{side}{k}"], te[f"y_{side}{k}"]
            arms = {}
            for name, xtr, xte in [("STATE", tr["S"], te["S"]),
                                   ("FRAME", tr["FR"], te["FR"])]:
                if side == "fut":                      # 前瞻两臂同给 plan
                    xtr = np.concatenate([xtr, tr["PLAN"][k]], 1)
                    xte = np.concatenate([xte, te["PLAN"][k]], 1)
                arms[name] = ridge_r2(xtr, ytr, xte, yte)
            se_c = ((te["y_now"] - yte) ** 2).sum(1)
            sst = ((yte - yte.mean(0)) ** 2).sum(1)
            lo, hi = paired_r2_ci(arms["STATE"][1], arms["FRAME"][1], sst)
            res[f"{side}{k}"] = {
                "r2_STATE": round(float(arms["STATE"][0]), 4),
                "r2_FRAME": round(float(arms["FRAME"][0]), 4),
                "r2_COPY": round(float(1 - se_c.sum() / sst.sum()), 4),
                "delta_ci": [round(lo, 4), round(hi, 4)]}
        print(f"[S7b] k={k} done", flush=True)
    # ── S5c 可控性 ──
    m = te["dxa"] >= DX_TH
    d = te["e_flip"][m] - te["e_true"][m]
    rng = np.random.default_rng(0)
    bs = [d[i].mean() for i in
          (rng.integers(0, len(d), len(d)) for _ in range(500))]
    c_lo, c_hi = np.percentile(bs, [2.5, 97.5])
    res["s5c"] = {"n": int(m.sum()), "d_err_mean": round(float(d.mean()), 5),
                  "ci": [round(float(c_lo), 5), round(float(c_hi), 5)]}

    # ── S7b 判据:horizon 跨 W1 对照 + control CI ──
    d_w4 = res["past10"]["r2_STATE"] - res["past10"]["r2_FRAME"]
    w1 = None
    if os.path.exists(args.w1):
        w1 = json.load(open(args.w1))
    if w1 and "past10" in w1:
        d_w1 = w1["past10"]["r2_STATE"] - w1["past10"]["r2_FRAME"]
        horizon_ok = (d_w4 - d_w1) > 0
        res["s7b_horizon"] = {
            "d_STATE_minus_FRAME_W4": round(d_w4, 4),
            "d_STATE_minus_FRAME_W1": round(d_w1, 4),
            "dd_W4_minus_W1": round(d_w4 - d_w1, 4)}
    else:
        horizon_ok = None
        res["s7b_horizon"] = {"d_STATE_minus_FRAME_W4": round(d_w4, 4),
                              "note": f"W1 基线缺失({args.w1}),horizon 半无法判"}
    control_ok = c_lo > 0
    both = (horizon_ok is True) and control_ok
    res["verdict_s7b_horizon"] = (
        "PASS" if horizon_ok else ("N/A" if horizon_ok is None else "FAIL")) + \
        f" (ΔΔ={res['s7b_horizon'].get('dd_W4_minus_W1')})"
    res["verdict_s7b_control"] = ("PASS" if control_ok else "FAIL") + \
        f" (S5c CI=[{c_lo:.5f},{c_hi:.5f}])"
    res["verdict_s7b"] = ("PASS" if both else "FAIL") + \
        " (horizon AND control 皆须)"
    print("[S7b]", json.dumps(res, ensure_ascii=False), flush=True)
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
