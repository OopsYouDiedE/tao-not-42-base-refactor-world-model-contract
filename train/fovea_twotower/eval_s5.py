# -*- coding: utf-8 -*-
"""S5:脑内世界正面度量(what's right)—— step2 §6,冻结 W1 塔,零训练。

三个正面属性(判据预登记于 step2 §6,均为"状态装了一个可用的世界"的直接证据,
而非消融式"去掉X掉多少"):
    S5a 记忆视界   — 从 STATE 线性重建 t−k 帧潜变量(3×3 池化),对比 FRAME(单帧);
                     注册点 k=10(1s):R²(STATE)−R²(FRAME) CI>0。
    S5b 前瞻视界   — 重建 t+k 帧,两臂**同给**已执行动作序列(plan);STATE 仍胜
                     = 状态含超越当前一瞥的世界信息且对未来有预测力。注册点 k=+10。
    S5c 可控性     — 同一前缀,末动作 dx 取真实 vs 翻符号,读塔对下帧首 patch 的
                     预测;真实动作的预测误差应显著小于翻转(限 |f(dx)|≥0.5 锚点)
                     = 预测是动作条件化的(模拟器),不是缓存回放。
辅助输出:k∈{2,5,10,20} 全曲线 + COPY 基线(直接拿 t 帧当预测)。

用法:PYTHONPATH=. python train/fovea_twotower/eval_s5.py
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
from train.fovea_twotower.eval_utils import pool9, ridge_r2, paired_r2_ci
from train.gaming500.dataset import Gaming500Dataset, N_MSG

DEV = "cuda"
DATA = "runs/data/g500_360p"
CKPT = "runs/ftt_w1/ckpt.pt"
KS = [2, 5, 10, 20]                                    # 帧;10Hz → 0.2/0.5/1/2s
T_ANCHOR = 43                                          # 64 帧窗:前 44 帧前缀,留 20 帧未来
DX_TH = 0.5                                            # 可控性锚点阈(featurize 域,同 eval_s4)
P = 81 + 1 + 1                                         # 帧块周期(vis|msg|act)


@torch.no_grad()
def collect(model, dino, dl, n_max):
    """每窗 1 锚点 t=T_ANCHOR。返回特征/目标/可控性误差(numpy)。"""
    t = T_ANCHOR
    S, FR, PLAN, Y = [], [], {k: [] for k in KS}, {k: {} for k in KS}
    E_true, E_flip, DXA = [], [], []
    n = 0
    for batch in dl:
        lat, act, msg = batch_to_stream_msg(batch, dino, DEV)
        B = lat.shape[0]
        # ── 特征臂 ──
        _, states = model.encode(lat[:, :t + 1], act[:, :t], msg[:, :t + 1],
                                 want_states=True)
        S.append(torch.cat([st["recurrent_state"].mean(2).flatten(1)
                            for st in states], 1).float().cpu())
        h_fr, _ = model.encode(lat[:, t:t + 1], act[:, :0], msg[:, t:t + 1])
        FR.append(h_fr.mean(1).float().cpu())
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
            errs.append(((pred - tgt0) ** 2).mean(-1).float().cpu())
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
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--out", default="runs/ftt_s5_360.json")
    args = p.parse_args()
    torch.manual_seed(0)
    ck = torch.load(args.ckpt, map_location=DEV)
    model = ContextTower(n_msg=N_MSG,
                         aux_msg=ck.get("args", {}).get("aux_msg", 0.0)
                         ).to(DEV).bfloat16().eval()
    model.load_state_dict(ck["model"])
    crop = ck.get("args", {}).get("crop", "center")
    dino = build_backbone(BackboneConfig(kind="dinov2"))[0].to(DEV).eval()
    mk = lambda split, sh: DataLoader(
        Gaming500Dataset(DATA, seq_len=64, img_size=126, stride=32,
                         crop_mode=crop, periph=True, split=split,
                         holdout_frac=0.1),
        batch_size=8, shuffle=sh, num_workers=8)
    tr = collect(model, dino, mk("train", True), 2500)
    te = collect(model, dino, mk("holdout", False), 1500)
    print(f"[S5] n_train={len(tr['dxa'])} n_test={len(te['dxa'])}", flush=True)

    res = {"n_train": len(tr["dxa"]), "n_test": len(te["dxa"])}
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
            se_c = ((te["y_now"] - yte) ** 2).sum(1)   # COPY 基线
            sst = ((yte - yte.mean(0)) ** 2).sum(1)
            lo, hi = paired_r2_ci(arms["STATE"][1], arms["FRAME"][1], sst)
            res[f"{side}{k}"] = {
                "r2_STATE": round(float(arms["STATE"][0]), 4),
                "r2_FRAME": round(float(arms["FRAME"][0]), 4),
                "r2_COPY": round(float(1 - se_c.sum() / sst.sum()), 4),
                "delta_ci": [round(lo, 4), round(hi, 4)]}
        print(f"[S5] k={k} done", flush=True)
    # ── S5c 可控性(holdout,|f(dx)|≥阈)──
    m = te["dxa"] >= DX_TH
    d = te["e_flip"][m] - te["e_true"][m]
    rng = np.random.default_rng(0)
    bs = [d[i].mean() for i in
          (rng.integers(0, len(d), len(d)) for _ in range(500))]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    res["s5c"] = {"n": int(m.sum()), "d_err_mean": round(float(d.mean()), 5),
                  "ci": [round(float(lo), 5), round(float(hi), 5)]}

    v = lambda ok, s: ("PASS " if ok else "FAIL ") + s
    res["verdict_s5a"] = v(res["past10"]["delta_ci"][0] > 0,
                           f"记忆视界 k=-10 ΔR² CI={res['past10']['delta_ci']}")
    res["verdict_s5b"] = v(res["fut10"]["delta_ci"][0] > 0,
                           f"前瞻视界 k=+10 ΔR² CI={res['fut10']['delta_ci']}")
    res["verdict_s5c"] = v(lo > 0, f"可控性 Δerr(flip-true) CI=[{lo:.5f},{hi:.5f}]")
    print("[S5]", json.dumps(res, ensure_ascii=False), flush=True)
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
