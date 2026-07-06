#!/usr/bin/env python3
"""预编译 VPT 承包商 clip → 冻结骨干 CLS 特征,存快盘,供快头 BC 免解码/免 DINO 前向直训。

动机:net/bc/policy.BCPolicy 的骨干**全程冻结**(no_grad),`encode_frames` 每帧只出
CLS 向量 [enc_dim=384];train_bc 的 `forward(feats, prev_act)` 直接吃 feats。故 DINO
特征是**固定**的——预编译一次即可从训练热循环里拿掉 mp4 解码 + DINO 前向,只剩小 trunk。

分盘(遵用户规则):原始 mp4 留数据盘 /data;编译后 feats 存 nvme 快盘(默认 runs/data/vpt_feat)。
体积:[T,384] fp16 ≈ 每段 ~5MB,几百段仅 ~GB 级。

与训练逐比特一致:用 `build_bc_policy(BCConfig)` 的**同一 backbone + 同一预处理**
(ImageNet 归一化、CLS token),img_size 与 train_bc 对齐(默认 128)。产物 npz:
  feats  fp16 [T, 384]        逐帧 CLS 特征
  action fp32 [T, action_dim] 逐帧动作(vpt_dataset._action_vec 契约,鼠标按 camera_scale 归一)
  task   str                  首帧任务文本
并写 manifest.json 记 backbone/img_size/camera_scale,训练侧据此校验缓存是否匹配。

增量:已存在同名 .npz 则跳过 → 可与下载并行、可反复重跑补齐。

用法(排在 W4c 之后跑,需 GPU;dinov3 首次拉权重需 HF token + SOCKS):
  export HF_TOKEN=... ALL_PROXY=socks5://127.0.0.1:2080 HTTPS_PROXY=socks5://127.0.0.1:2080
  PYTHONPATH=. python tests/encode_vpt_feats.py \
      --data-dir /data/vpt_minecraft --out runs/data/vpt_feat --img-size 128 --backbone dinov3
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np
import torch

from net.bc import build_bc_policy, BCConfig
from net.config import BackboneConfig
from train.minecraft.vpt_action import CAMERA_SCALE
from train.minecraft.vpt_dataset import _action_vec, VPT_KEYS


def find_clips(data_dir):
    """递归找所有成对 (mp4, jsonl)。"""
    pairs = []
    for mp4 in sorted(glob.glob(os.path.join(data_dir, "**", "*.mp4"), recursive=True)):
        jp = mp4[:-4] + ".jsonl"
        if os.path.exists(jp):
            pairs.append((mp4, jp))
    return pairs


def decode_frames(mp4, img_size):
    """整段解码 → uint8 [T,3,H,W](RGB,resize 到 img_size)。"""
    cap = cv2.VideoCapture(mp4)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        fr = cv2.resize(fr, (img_size, img_size), interpolation=cv2.INTER_AREA)
        frames.append(fr)
    cap.release()
    if not frames:
        return None
    return np.stack(frames)  # [T,H,W,3] uint8


def load_actions(jsonl, camera_scale):
    """逐帧动作 → [T, action_dim] fp32(_action_vec 契约)。返回 (acts, task)。"""
    acts, task = [], None
    with open(jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            try:
                d = json.loads(line) if line else {}
            except ValueError:
                d = {}
            if i == 0:
                task = d.get("task")
            acts.append(_action_vec(d, camera_scale).numpy())
    return np.stack(acts).astype(np.float32), task


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/data/vpt_minecraft", help="原始 mp4+jsonl(数据盘)")
    p.add_argument("--out", default="runs/data/vpt_feat", help="feats 输出(快盘 nvme)")
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--backbone", choices=["dinov3", "dinov2"], default="dinov3")
    p.add_argument("--camera-scale", type=float, default=CAMERA_SCALE)
    p.add_argument("--batch", type=int, default=256, help="DINO 前向的帧批大小")
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    policy = build_bc_policy(BCConfig(backbone=BackboneConfig(kind=args.backbone))).to(dev).eval()
    enc_dim = policy.enc_dim

    manifest = {"backbone": args.backbone, "img_size": args.img_size,
                "camera_scale": args.camera_scale, "enc_dim": enc_dim,
                "action_keys": VPT_KEYS}
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    clips = find_clips(args.data_dir)
    print(f"[encode] {len(clips)} clips in {args.data_dir} | backbone={args.backbone} "
          f"img={args.img_size} enc_dim={enc_dim} → {args.out}", flush=True)
    done = skip = fail = 0
    for mp4, jsonl in clips:
        name = os.path.splitext(os.path.basename(mp4))[0]
        outp = os.path.join(args.out, name + ".npz")
        if os.path.exists(outp):
            skip += 1
            continue
        t0 = time.time()
        imgs = decode_frames(mp4, args.img_size)
        acts, task = load_actions(jsonl, args.camera_scale)
        if imgs is None or len(acts) == 0:
            fail += 1
            print(f"[encode] ✗ {name} 解码空", flush=True)
            continue
        T = min(len(imgs), len(acts))
        imgs, acts = imgs[:T], acts[:T]
        # 逐批过冻结骨干取 CLS
        feats = np.empty((T, enc_dim), np.float16)
        for s in range(0, T, args.batch):
            chunk = torch.from_numpy(imgs[s:s + args.batch]).to(dev)
            chunk = chunk.permute(0, 3, 1, 2).float() / 255.0     # [b,3,H,W]
            f = policy.encode_frames(chunk.unsqueeze(1))[:, 0]     # [b,enc_dim]
            feats[s:s + chunk.shape[0]] = f.float().cpu().numpy().astype(np.float16)
        np.savez(outp, feats=feats, action=acts, task=task or "")
        done += 1
        print(f"[encode] ✓ {name} T={T} {(time.time()-t0):.1f}s "
              f"({feats.nbytes/1e6:.1f}MB) [{done} done/{skip} skip/{fail} fail]", flush=True)
    tot = sum(os.path.getsize(f) for f in glob.glob(os.path.join(args.out, "*.npz")))
    print(f"[encode] DONE {done} 新编/{skip} 跳过/{fail} 失败;{args.out} 共 {tot/1e9:.2f}GB", flush=True)


if __name__ == "__main__":
    main()
