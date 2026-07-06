# -*- coding: utf-8 -*-
"""S3a 终判:R2(播种) vs B1(零状态) 的免阈值配对比较。

f1_attack 在 6000 步内贴地(稀有键+0.5硬阈值),不宜做主判据;
改用连续输出的 frame 级 ROC-AUC(attack 位),并对 Δ 做窗口级配对 bootstrap:
两模型在同一 holdout 窗口、同一采样噪声(固定 generator)下出 chunk,
Δauc 的 95% CI 不含 0 才算显著。同时报 F1(原口径)与 onset 子集 AUC。
"""
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from net.backbone import build_backbone
from net.config import BackboneConfig
from net.fovea_twotower import ActionTower, ContextTower
from train.fovea_twotower.eval_utils import auc
from train.fovea_twotower.train_r2 import H, prep
from train.gaming500.dataset import Gaming500Dataset, KEY_NAMES

I_ATT = 2 + KEY_NAMES.index("key_attack")              # featurize 内 attack 下标
DEV = "cuda"
DATA = "runs/data/g500_360p"
CTX = "runs/ftt_r1_360/ckpt.pt"


@torch.no_grad()
def gather(model, ctx, dino, dl, mode, n_max=2000):
    """→ pred [N,H], target [N,H](attack 位), pred_k/targ_k [N,H,20], past_attack [N]"""
    gen = torch.Generator(DEV).manual_seed(1234)
    P, T, PK, TK, PA = [], [], [], [], []
    n = 0
    for batch in dl:
        lat_now, states, z1 = prep(batch, dino, ctx, mode, DEV)
        pred = model.sample(lat_now, seed=states, steps=4, generator=gen)
        P.append(pred[..., I_ATT].float().cpu())
        T.append((z1[..., I_ATT] > 0.5).float().cpu())
        PK.append(pred[..., 2:22].float().cpu())
        TK.append((z1[..., 2:22] > 0.5).float().cpu())
        t = batch["keys"].shape[1] - 1 - H
        PA.append(batch["keys"][:, t - 3:t, KEY_NAMES.index("key_attack")]
                  .any(1).float())
        n += z1.shape[0]
        if n >= n_max:
            break
    c = lambda xs: torch.cat(xs).numpy()
    return c(P), c(T), c(PK), c(TK), c(PA)


def keys_macro_auc(TK, PK, idx=None):
    """20 键宏平均 AUC(仅取两类齐全的键);idx=窗口下标(bootstrap 用)。"""
    if idx is not None:
        TK, PK = TK[idx], PK[idx]
    vals = []
    for k in range(TK.shape[-1]):
        a = auc(TK[..., k], PK[..., k])
        if np.isfinite(a):
            vals.append(a)
    return float(np.mean(vals)) if vals else float("nan")


def load_tower(path, ctx):
    m = ActionTower(horizon=H).to(DEV).bfloat16()
    m.load_state_dict(torch.load(path, map_location=DEV)["model"])
    return m.eval()


def main():
    torch.manual_seed(0)
    dino = build_backbone(BackboneConfig(kind="dinov2"))[0].to(DEV).eval()
    ctx = ContextTower().to(DEV).bfloat16().eval()
    ctx.load_state_dict(torch.load(CTX, map_location=DEV)["model"])
    mk = lambda: DataLoader(
        Gaming500Dataset(DATA, seq_len=64, img_size=126, stride=32,
                         crop_mode="center", split="holdout", holdout_frac=0.1),
        batch_size=8, shuffle=False, num_workers=6)
    r2 = load_tower("runs/ftt_r2_seed_360/ckpt.pt", ctx)
    b1 = load_tower("runs/ftt_r2_zero_360/ckpt.pt", ctx)
    ms = load_tower("runs/ftt_r2_mis_360/ckpt.pt", ctx)
    # 同一数据顺序(shuffle=False)+ 同一采样 generator → 逐窗口配对
    Pr, Tr, PKr, TKr, PAr = gather(r2, ctx, dino, mk(), mode=1)
    Pb, Tb, PKb, TKb, _ = gather(b1, ctx, dino, mk(), mode=0)
    Pm, Tm, PKm, TKm, _ = gather(ms, ctx, dino, mk(), mode=2)
    assert Tr.shape == Tb.shape and np.allclose(Tr, Tb), "窗口未对齐"
    assert np.allclose(Tr, Tm), "B1.5 窗口未对齐"

    res = {"n_windows": int(Tr.shape[0]),
           "attack_frame_rate": round(float(Tr.mean()), 4),
           "auc_attack_R2": round(auc(Tr, Pr), 4),
           "auc_attack_B1": round(auc(Tb, Pb), 4)}
    # 窗口级配对 bootstrap
    rng = np.random.default_rng(0)
    N = Tr.shape[0]
    ds = []
    for _ in range(1000):
        i = rng.integers(0, N, N)
        ds.append(auc(Tr[i], Pr[i]) - auc(Tb[i], Pb[i]))
    ds = np.array([d for d in ds if np.isfinite(d)])
    lo, hi = np.percentile(ds, [2.5, 97.5])
    res["delta_auc"] = round(res["auc_attack_R2"] - res["auc_attack_B1"], 4)
    res["delta_ci95"] = [round(float(lo), 4), round(float(hi), 4)]
    # keys 宏平均 AUC(播种是否传递了可用历史——不区分世界记忆/动作缓冲)
    res["auc_keys_R2"] = round(keys_macro_auc(TKr, PKr), 4)
    res["auc_keys_B1"] = round(keys_macro_auc(TKb, PKb), 4)
    dk = []
    for _ in range(300):
        i = rng.integers(0, N, N)
        dk.append(keys_macro_auc(TKr, PKr, i) - keys_macro_auc(TKb, PKb, i))
    dk = np.array([d for d in dk if np.isfinite(d)])
    klo, khi = np.percentile(dk, [2.5, 97.5])
    res["delta_keys_auc"] = round(res["auc_keys_R2"] - res["auc_keys_B1"], 4)
    res["delta_keys_ci95"] = [round(float(klo), 4), round(float(khi), 4)]
    # onset 子集(过去3帧未开火的窗口)
    m = PAr == 0
    res["n_onset"] = int(m.sum())
    res["auc_onset_R2"] = round(auc(Tr[m], Pr[m]), 4)
    res["auc_onset_B1"] = round(auc(Tb[m], Pb[m]), 4)
    res["verdict_s3a"] = ("PASS" if lo > 0 else "FAIL") + \
        f" (Δauc={res['delta_auc']:+.4f}, CI95=[{lo:+.4f},{hi:+.4f}])"
    # B1.5 错配归因:R2(匹配) − B1.5(错配) 的 keys/attack 配对差
    res["auc_attack_MIS"] = round(auc(Tm, Pm), 4)
    res["auc_keys_MIS"] = round(keys_macro_auc(TKm, PKm), 4)
    dm = []
    for _ in range(300):
        i = rng.integers(0, N, N)
        dm.append(keys_macro_auc(TKr, PKr, i) - keys_macro_auc(TKm, PKm, i))
    dm = np.array([d for d in dm if np.isfinite(d)])
    mlo, mhi = np.percentile(dm, [2.5, 97.5])
    res["delta_keys_vs_mis"] = round(res["auc_keys_R2"] - res["auc_keys_MIS"], 4)
    res["delta_keys_vs_mis_ci95"] = [round(float(mlo), 4), round(float(mhi), 4)]
    res["verdict_b15"] = ("内容归因成立" if mlo > 0 else "统计先验解释未排除") + \
        f" (Δkeys_R2-MIS={res['delta_keys_vs_mis']:+.4f}, CI95=[{mlo:+.4f},{mhi:+.4f}])"
    print("[S3a-360]", json.dumps(res, ensure_ascii=False), flush=True)
    json.dump(res, open("runs/ftt_s3a_360.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
