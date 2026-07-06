#!/usr/bin/env python3
"""反事实文本操纵探针(S6 式,低方差):同一批 held-out 状态,只翻转指令 emb,
量策略输出的差。rollout 挥击计数 n=6 噪声太大;这里在**固定状态**上比 mine vs still,
把采样噪声彻底剔除 → 干净读"文字是否操纵执行"。

对每条 held-out feats 序列,跑 policy(feats, prev, mine) 与 policy(feats, prev, still),
取 attack 键 logit→sigmoid=P(挖) 与相机期望,报:
  ΔP(attack) = mean[P(attack|mine) − P(attack|still)]  (>0 且逐序一致 = 操纵成立)
  可导性 = 逐序 P(attack|mine)>P(attack|still) 的比例(bootstrap CI 下界>0.5 过)

用法:
  PYTHONPATH=. python tests/integration/cf_probe.py --ckpt runs/rest_prog/r2/final.pt \
      --feats_dir runs/rest_prog/r2_gate/feats --a mine --b still
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from net.bc import BCConfig
from net.config import BackboneConfig
from net.bc import TextCondPolicy
from tests.integration.collect_s8 import V2_KEYS
from train.minecraft.vpt_action import ACTION_DIM, CAMERA_BINS, N_MOUSE

I_ATTACK = V2_KEYS.index("attack")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--instr_emb", default="runs/ftt_instr/instr_emb.pt")
    p.add_argument("--feats_dir", required=True, help="held-out feats npz 目录(状态分布)")
    p.add_argument("--a", default="mine"); p.add_argument("--b", default="still")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ie = torch.load(args.instr_emb); id2idx = {i: k for k, i in enumerate(ie["ids"])}
    ea = ie["emb"][id2idx[args.a]].view(1, -1).to(device).float()
    eb = ie["emb"][id2idx[args.b]].view(1, -1).to(device).float()
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cs = ck.get("cfg", {})
    cfg = BCConfig(backbone=BackboneConfig(kind="dinov3"), d=cs.get("d", 384),
                   heads=cs.get("heads", 6), layers=cs.get("layers", 4), dropout=0.0,
                   max_len=max(128, args.seq_len), action_dim=ACTION_DIM, n_mouse=N_MOUSE,
                   camera_bins=CAMERA_BINS)
    from train.fovea_twotower.rest_update import _Stub
    policy = TextCondPolicy(cfg, injected_backbone=_Stub(384)).to(device).eval()
    policy.load_state_dict(ck.get("policy", ck), strict=False)

    files = sorted(glob.glob(os.path.join(args.feats_dir, "*.npz")))
    pa_seq, pb_seq = [], []
    with torch.no_grad():
        for f in files:
            z = np.load(f)
            feats = torch.from_numpy(z["feats"][:args.seq_len].astype(np.float32)).unsqueeze(0).to(device)
            act = torch.from_numpy(z["action"][:args.seq_len].astype(np.float32)).unsqueeze(0).to(device)
            prev = torch.zeros_like(act); prev[:, 1:] = act[:, :-1]
            _, key_a = policy(feats, prev, ea)
            _, key_b = policy(feats, prev, eb)
            pa = key_a[0, :, I_ATTACK].float().sigmoid().mean().item()
            pb = key_b[0, :, I_ATTACK].float().sigmoid().mean().item()
            pa_seq.append(pa); pb_seq.append(pb)
    pa_seq, pb_seq = np.array(pa_seq), np.array(pb_seq)
    d = pa_seq - pb_seq
    win = float((d > 0).mean())
    rng = np.random.default_rng(0)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    res = {"ckpt": args.ckpt, "n_seq": len(files), "a": args.a, "b": args.b,
           "P_attack_a": round(float(pa_seq.mean()), 4), "P_attack_b": round(float(pb_seq.mean()), 4),
           "delta_P_attack": round(float(d.mean()), 4), "delta_ci": [round(float(lo), 4), round(float(hi), 4)],
           "steer_consistency": round(win, 3),
           "verdict": f"{'PASS' if lo > 0 else 'FAIL'} (ΔP(attack)={d.mean():+.4f} CI=[{lo:+.4f},{hi:+.4f}] 一致率={win:.2f})"}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if args.out:
        json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
