# -*- coding: utf-8 -*-
"""S6 终判(step3 指令总线):快脑服不服从钉住的文本指令 token。

同一 holdout、同一初始噪声(逐批同种子),每窗 5 个条件配对采样:
    FREE("act freely")/ TRUE(事后真指令)/ LEFT / RIGHT / FIRE。
预登记判据:
    S6a 通道读取   — 转向窗 dx 方向 AUC:TRUE−FREE 配对 Δ CI>0(指令泄露未来,
                     只证"token 被读",数值大小不作卖点);
    S6b 反事实服从 — 实际右转窗喂 turn left(及镜像),服从率 =
                     sign(预测dx和)==指令方向 的比例;bootstrap CI 下界>0.5 过。
                     辅:可导性 Δ = P(预测右|RIGHT) − P(预测右|LEFT) 全窗 CI>0;
    S6c 特异性     — 转向指令不应移动开火预测:|attack AUC(LEFT)−(FREE)| 报告;
                     开火可导性 = 无开火窗 mean pred_attack (FIRE)−(FREE) CI>0。
"""
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from net.backbone import build_backbone
from net.config import BackboneConfig
from net.fovea_twotower import ActionTower, ContextTower
from train.fovea_twotower.eval_utils import auc
from train.fovea_twotower.train_w2 import H, prep
from train.gaming500.dataset import Gaming500Dataset, KEY_NAMES, N_MSG

DEV = "cuda"
DATA = "runs/data/g500_360p"
CTX = "runs/ftt_w1/ckpt.pt"
C1 = "runs/ftt_c1/ckpt.pt"
EMB = "runs/ftt_cmd/cmd_emb.pt"
DX_TH = 0.5
I_ATT = 2 + KEY_NAMES.index("key_attack")
LEFT, RIGHT, FIRE, FWD, JUMP, FREE = range(6)
CONDS = {"FREE": FREE, "LEFT": LEFT, "RIGHT": RIGHT, "FIRE": FIRE}


def true_cmd(z1):
    """每窗事后真指令(优先转向,其次开火,兜底 FREE)——S6a 用。"""
    dxs = z1[..., 0].sum(1).float()
    fire = (z1[..., I_ATT] > 0.5).any(1)
    ids = torch.full((z1.shape[0],), FREE)
    ids[fire.cpu()] = FIRE
    ids[(dxs < -DX_TH).cpu()] = LEFT
    ids[(dxs > DX_TH).cpu()] = RIGHT
    return ids


@torch.no_grad()
def gather(model, ctx, dino, dl, emb, n_max=2000):
    sgen = torch.Generator().manual_seed(777)
    out = {k: {"dx": [], "att": []} for k in list(CONDS) + ["TRUE"]}
    DXT, ATT = [], []
    n, bi = 0, 0
    for batch in dl:
        lat_now, states, z1, _ = prep(batch, dino, ctx, 1, sgen, DEV)
        DXT.append(z1[..., 0].sum(1).float().cpu())
        ATT.append((z1[..., I_ATT] > 0.5).any(1).float().cpu())
        cmds = dict(CONDS)
        for name, cid in cmds.items():
            c = emb[torch.full((z1.shape[0],), cid)].to(DEV)
            gen = torch.Generator(DEV).manual_seed(9000 + bi)   # 条件间同噪声
            pred = model.sample(lat_now, seed=states, steps=4, generator=gen, cmd=c)
            out[name]["dx"].append(pred[..., 0].sum(1).float().cpu())
            out[name]["att"].append(pred[..., I_ATT].max(1).values.float().cpu())
        c = emb[true_cmd(z1)].to(DEV)
        gen = torch.Generator(DEV).manual_seed(9000 + bi)
        pred = model.sample(lat_now, seed=states, steps=4, generator=gen, cmd=c)
        out["TRUE"]["dx"].append(pred[..., 0].sum(1).float().cpu())
        out["TRUE"]["att"].append(pred[..., I_ATT].max(1).values.float().cpu())
        n += z1.shape[0]
        bi += 1
        if n >= n_max:
            break
    c = lambda xs: torch.cat(xs).numpy()
    return ({k: {m: c(v[m]) for m in v} for k, v in out.items()}, c(DXT), c(ATT))


def boot_ci(fn, N, boot=500):
    rng = np.random.default_rng(0)
    vs = [fn(rng.integers(0, N, N)) for _ in range(boot)]
    lo, hi = np.percentile(vs, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    torch.manual_seed(0)
    dino = build_backbone(BackboneConfig(kind="dinov2"))[0].to(DEV).eval()
    ck = torch.load(CTX, map_location=DEV)
    ctx = ContextTower(n_msg=N_MSG).to(DEV).bfloat16().eval()
    ctx.load_state_dict(ck["model"])
    crop = ck.get("args", {}).get("crop", "center")
    emb = torch.load(EMB)["emb"].bfloat16()
    model = ActionTower(horizon=H, n_cmd=emb.shape[1]).to(DEV).bfloat16()
    model.load_state_dict(torch.load(C1, map_location=DEV)["model"])
    model.eval()
    dl = DataLoader(
        Gaming500Dataset(DATA, seq_len=64, img_size=126, stride=32,
                         crop_mode=crop, periph=True, split="holdout",
                         holdout_frac=0.1),
        batch_size=8, shuffle=False, num_workers=8)
    out, dxt, att = gather(model, ctx, dino, dl, emb)
    turn = np.abs(dxt) > DX_TH
    y_dir = (dxt > 0).astype(float)
    res = {"n": len(dxt), "n_turn": int(turn.sum())}

    # S6a 通道读取:转向窗 TRUE vs FREE 方向 AUC
    sub = np.nonzero(turn)[0]
    a_t, a_f = out["TRUE"]["dx"][sub], out["FREE"]["dx"][sub]
    y = y_dir[sub]
    lo, hi = boot_ci(lambda i: auc(y[i], a_t[i]) - auc(y[i], a_f[i]), len(sub))
    res["s6a"] = {"auc_TRUE": round(auc(y, a_t), 4), "auc_FREE": round(auc(y, a_f), 4),
                  "delta_ci": [round(lo, 4), round(hi, 4)]}

    # S6b 反事实服从:实际右转喂 LEFT + 实际左转喂 RIGHT
    cf = np.concatenate([np.sign(out["LEFT"]["dx"][dxt > DX_TH]) < 0,
                         np.sign(out["RIGHT"]["dx"][dxt < -DX_TH]) > 0]).astype(float)
    lo_o, hi_o = boot_ci(lambda i: cf[i].mean(), len(cf))
    # 可导性(全窗):P(预测右|RIGHT) − P(预测右|LEFT)
    pr_r = (out["RIGHT"]["dx"] > 0).astype(float)
    pr_l = (out["LEFT"]["dx"] > 0).astype(float)
    lo_s, hi_s = boot_ci(lambda i: pr_r[i].mean() - pr_l[i].mean(), len(pr_r))
    res["s6b"] = {"n_cf": len(cf), "obedience": round(float(cf.mean()), 4),
                  "obedience_ci": [round(lo_o, 4), round(hi_o, 4)],
                  "steer_delta": round(float(pr_r.mean() - pr_l.mean()), 4),
                  "steer_ci": [round(lo_s, 4), round(hi_s, 4)],
                  "follow_data_FREE": round(float(
                      ((out["FREE"]["dx"] > 0) == (dxt > 0))[turn].mean()), 4)}

    # S6c 特异性 + 开火可导性
    res["s6c"] = {
        "auc_att_FREE": round(auc(att, out["FREE"]["att"]), 4),
        "auc_att_LEFT": round(auc(att, out["LEFT"]["att"]), 4)}
    noatt = np.nonzero(att < 0.5)[0]
    d_fire = out["FIRE"]["att"][noatt] - out["FREE"]["att"][noatt]
    lo_f, hi_f = boot_ci(lambda i: d_fire[i].mean(), len(d_fire))
    res["s6c"]["fire_steer"] = round(float(d_fire.mean()), 4)
    res["s6c"]["fire_steer_ci"] = [round(lo_f, 4), round(hi_f, 4)]

    v = lambda ok, s: ("PASS " if ok else "FAIL ") + s
    res["verdict_s6a"] = v(res["s6a"]["delta_ci"][0] > 0,
                           f"Δauc CI={res['s6a']['delta_ci']}")
    res["verdict_s6b"] = v(lo_o > 0.5,
                           f"服从率={cf.mean():.3f} CI=[{lo_o:.3f},{hi_o:.3f}]")
    res["verdict_s6c_fire"] = v(lo_f > 0, f"fire_steer CI=[{lo_f:.4f},{hi_f:.4f}]")
    print("[S6]", json.dumps(res, ensure_ascii=False), flush=True)
    json.dump(res, open("runs/ftt_s6_360.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
