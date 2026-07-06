#!/usr/bin/env python3
"""同域 BC 一票否决实验·前端:S8 采集的强策略轨迹 → 快头 BC 特征(train_fasthead 契约)。

命题(预登记,先于结果):快头形态(冻结 dinov3 CLS + 小时序头)能否**表达**C2 采矿技能。
训练数据 = collect_s8 强策略轨迹(policy_strong=1,易+难起点同一行为策略);
评测 = gate_c2 放回同款房间 rollout。判据:易房间 20 局 score>0 ≥15%(老师 42% 的约 1/3)
= 执行接口通过;全 0 = 接口证伪(CLS 特征/动作解码是瓶颈,与数据量无关)。

契约(与 encode_g500_feats 逐字段一致,train_fasthead 直接吃):
  feats  fp16 [T,384]  dinov3 CLS(frames 126→128 resize,同 gate crop128 口径)
  action fp32 [T,22]   [dx/CAMERA_SCALE, dy/CAMERA_SCALE, keys×20];末行 0
用法:
  PYTHONPATH=. python tests/encode_c2_feats.py --raw runs/data/s8_full --out runs/data/c2_bc_feat
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch

from net.bc import BCConfig, build_bc_policy
from net.config import BackboneConfig
from train.minecraft.vpt_action import CAMERA_SCALE


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="runs/data/s8_full")
    p.add_argument("--out", default="runs/data/c2_bc_feat")
    p.add_argument("--strong-only", action=argparse.BooleanOptionalAction, default=True,
                   help="只取 policy_strong=1(技能表达测试的老师示范)")
    p.add_argument("--batch", type=int, default=256)
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy = build_bc_policy(BCConfig(backbone=BackboneConfig(kind="dinov3"))).to(dev).eval()
    os.makedirs(args.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.raw, "*.npz")))
    done = skip = 0
    for fp in files:
        z = np.load(fp, allow_pickle=True)
        if args.strong_only and int(z["policy_strong"]) != 1:
            continue
        outp = os.path.join(args.out, os.path.basename(fp))
        if os.path.exists(outp):
            skip += 1
            continue
        frames = z["frames"]                              # [T,3,126,126] u8
        T = frames.shape[0]
        imgs = np.empty((T, 128, 128, 3), np.uint8)
        for i in range(T):                                # 126→128,同 gate crop128 口径
            imgs[i] = cv2.resize(frames[i].transpose(1, 2, 0), (128, 128),
                                 interpolation=cv2.INTER_AREA)
        feats = np.empty((T, policy.enc_dim), np.float16)
        for s in range(0, T, args.batch):
            chunk = torch.from_numpy(imgs[s:s + args.batch]).to(dev)
            chunk = chunk.permute(0, 3, 1, 2).float() / 255.0
            fv = policy.encode_frames(chunk.unsqueeze(1))[:, 0]
            feats[s:s + chunk.shape[0]] = fv.float().cpu().numpy().astype(np.float16)
        act = np.zeros((T, 22), np.float32)               # 末行 0(encode_g500 同约定)
        act[:-1, 0] = np.clip(z["dx"] / CAMERA_SCALE, -1.0, 1.0)
        act[:-1, 1] = np.clip(z["dy"] / CAMERA_SCALE, -1.0, 1.0)
        act[:-1, 2:] = z["keys"]
        np.savez(outp, feats=feats, action=act,
                 score=z["score"], start_hard=z["start_hard"])
        done += 1
        if done % 20 == 0:
            print(f"[c2feat] {done} done", flush=True)
    print(f"[c2feat] DONE {done} 新编/{skip} 跳过 → {args.out}", flush=True)


if __name__ == "__main__":
    main()
