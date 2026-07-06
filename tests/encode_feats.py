#!/usr/bin/env python3
"""预编译轨迹/视频 → 冻结骨干 CLS 特征,存快盘,供快头 BC 免解码/免 DINO 前向直训。

三源合一(--source-type 分发,共享同一 DINO 编码核与 npz 契约):
  c2    S8 采集的 C2 采矿轨迹(runs/data/s8_full/*.npz 的 frames[T,3,126,126]);
        额外存 score/start_hard;--strong-only 只取 policy_strong=1 老师示范。
  g500  gaming500-MC 编码档 HDF5 分片(encode_gaming500_hdf5 产出);15Hz 图 × 30Hz 事件
        区间聚合动作(dx/dy 求和、keys 位 OR、gui max);额外存 dx_px/dy_px/gui/task。
  vpt   VPT 承包商 clip(递归 mp4+jsonl);_action_vec 契约;额外存 task。

统一 npz 契约(train_fasthead 直接吃,训练侧不分叉):
  feats  fp16 [T, enc_dim]   逐帧 CLS(BCPolicy.encode_frames,同一 backbone + ImageNet 预处理)
  action fp32 [T, 22]        [dx/camera_scale, dy/camera_scale, keys×20];末帧动作置 0
BCPolicy 骨干全程冻结(no_grad),故 DINO 特征固定,预编一次即从训练热路径拿掉解码 + 前向。
增量:已存在同名 .npz 跳过 → 可与下载并行、反复重跑补齐。g500/vpt 另写 manifest.json 供训练侧校验。

用法(需 GPU;dinov3 首次拉权重需 HF token + SOCKS):
  PYTHONPATH=. python tests/encode_feats.py --source-type c2  --src runs/data/s8_full --out runs/data/c2_bc_feat
  PYTHONPATH=. python tests/encode_feats.py --source-type g500 --src /data/gaming500_mc_h5 --out runs/data/g500_mc_feat
  PYTHONPATH=. python tests/encode_feats.py --source-type vpt  --src /data/vpt_minecraft --out runs/data/vpt_feat
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np
import torch

from net.bc import BCConfig, build_bc_policy
from net.config import BackboneConfig
from train.minecraft.vpt_action import CAMERA_SCALE

_DEFAULT_SRC = {"c2": "runs/data/s8_full", "g500": "/data/gaming500_mc_h5",
                "vpt": "/data/vpt_minecraft"}


def _resize_rgb(im_hwc, img_size):
    """任意 RGB [H,W,3] → [img_size,img_size,3] uint8(INTER_AREA)。"""
    return cv2.resize(im_hwc, (img_size, img_size), interpolation=cv2.INTER_AREA)


# ── 三种数据源:各 yield dict(name, imgs[T,H,W,3]u8, action[T,22]f32, extra) ──
def iter_c2(args):
    for fp in sorted(glob.glob(os.path.join(args.src, "*.npz"))):
        z = np.load(fp, allow_pickle=True)
        if args.strong_only and int(z["policy_strong"]) != 1:
            continue
        frames = z["frames"]                               # [T,3,126,126] u8
        T = frames.shape[0]
        imgs = np.stack([_resize_rgb(frames[i].transpose(1, 2, 0), args.img_size)
                         for i in range(T)])
        act = np.zeros((T, 22), np.float32)                # 末行 0
        act[:-1, 0] = np.clip(z["dx"] / args.camera_scale, -1.0, 1.0)
        act[:-1, 1] = np.clip(z["dy"] / args.camera_scale, -1.0, 1.0)
        act[:-1, 2:] = z["keys"]
        yield dict(name=os.path.basename(fp)[:-4], imgs=imgs, action=act,
                   extra={"score": z["score"], "start_hard": z["start_hard"]})


def iter_g500(args):
    import h5py
    from train.gaming500.dataset import unpack_keys, KEY_NAMES
    from train.minecraft.vpt_dataset import VPT_KEYS
    assert KEY_NAMES == VPT_KEYS, "gaming500 KEY_NAMES 与 VPT_KEYS 位序不一致,动作契约失效"
    parts = [(f"{game}/{name}", shp)
             for shp in sorted(glob.glob(os.path.join(args.src, "shard_*.h5")))
             for game in h5py.File(shp, "r") for name in h5py.File(shp, "r")[game]]
    if args.limit:
        parts = parts[:args.limit]
    for key, shp in parts:
        with h5py.File(shp, "r") as f:
            g = f[key]
            task = json.loads(g.attrs.get("meta_json", "{}")).get("title", "") \
                or str(g.attrs.get("task", ""))
            fidx = g["frame_idx"][:].astype(np.int64)
            dx_f, dy_f, keys_f, gui_f = g["dx"][:], g["dy"][:], g["keys"][:], g["gui"][:]
            T, M = len(fidx), len(dx_f)
            dx = np.zeros(T, np.float32); dy = np.zeros(T, np.float32)
            keys = np.zeros((T, len(VPT_KEYS)), np.uint8); gui = np.zeros(T, np.uint8)
            for j in range(T - 1):                         # 图 j→j+1 区间聚合(同 Gaming500Dataset)
                lo, hi = min(fidx[j] + 1, M), min(fidx[j + 1] + 1, M)
                if lo >= hi:
                    continue
                dx[j] = dx_f[lo:hi].sum(); dy[j] = dy_f[lo:hi].sum()
                keys[j] = np.bitwise_or.reduce(unpack_keys(keys_f[lo:hi]), axis=0)
                gui[j] = gui_f[lo:hi].max()
            act = np.zeros((T, 2 + len(VPT_KEYS)), np.float32)
            act[:, 0] = np.clip(dx / args.camera_scale, -1.0, 1.0)
            act[:, 1] = np.clip(dy / args.camera_scale, -1.0, 1.0)
            act[:, 2:] = keys
            imgs = np.stack([_resize_rgb(
                cv2.cvtColor(cv2.imdecode(np.frombuffer(buf, np.uint8),
                             cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), args.img_size)
                for buf in g["jpeg"]])
        yield dict(name=key.split("/")[-1], imgs=imgs, action=act,
                   extra={"dx_px": dx, "dy_px": dy, "gui": gui, "task": task})


def iter_vpt(args):
    from train.minecraft.vpt_dataset import _action_vec
    pairs = [(mp4, mp4[:-4] + ".jsonl")
             for mp4 in sorted(glob.glob(os.path.join(args.src, "**", "*.mp4"), recursive=True))
             if os.path.exists(mp4[:-4] + ".jsonl")]
    for mp4, jsonl in pairs:
        cap = cv2.VideoCapture(mp4)
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(_resize_rgb(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), args.img_size))
        cap.release()
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
                acts.append(_action_vec(d, args.camera_scale).numpy())
        if not frames or not acts:
            print(f"[encode] ✗ {os.path.basename(mp4)} 解码空", flush=True)
            continue
        T = min(len(frames), len(acts))
        yield dict(name=os.path.splitext(os.path.basename(mp4))[0],
                   imgs=np.stack(frames[:T]),
                   action=np.stack(acts[:T]).astype(np.float32),
                   extra={"task": task or ""})


_ITER = {"c2": iter_c2, "g500": iter_g500, "vpt": iter_vpt}


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-type", choices=list(_ITER), required=True)
    p.add_argument("--src", default=None, help="输入路径(缺省按 source-type 取默认)")
    p.add_argument("--out", required=True, help="feats 输出(快盘 nvme)")
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--backbone", choices=["dinov3", "dinov2"], default="dinov3")
    p.add_argument("--camera-scale", type=float, default=CAMERA_SCALE)
    p.add_argument("--batch", type=int, default=256, help="DINO 前向的帧批大小")
    p.add_argument("--strong-only", action=argparse.BooleanOptionalAction, default=True,
                   help="[c2] 只取 policy_strong=1 老师示范")
    p.add_argument("--limit", type=int, default=0, help="[g500] >0 只编前 N 段(冒烟)")
    args = p.parse_args()
    args.src = args.src or _DEFAULT_SRC[args.source_type]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    policy = build_bc_policy(BCConfig(backbone=BackboneConfig(kind=args.backbone))).to(dev).eval()
    enc_dim = policy.enc_dim
    if args.source_type in ("g500", "vpt"):                # 训练侧据此校验缓存匹配
        from train.minecraft.vpt_dataset import VPT_KEYS
        json.dump({"backbone": args.backbone, "img_size": args.img_size,
                   "camera_scale": args.camera_scale, "enc_dim": enc_dim,
                   "action_keys": VPT_KEYS, "source": os.path.abspath(args.src)},
                  open(os.path.join(args.out, "manifest.json"), "w"),
                  indent=2, ensure_ascii=False)

    print(f"[encode] source={args.source_type} src={args.src} backbone={args.backbone} "
          f"img={args.img_size} enc_dim={enc_dim} → {args.out}", flush=True)
    done = skip = 0
    for clip in _ITER[args.source_type](args):
        outp = os.path.join(args.out, clip["name"] + ".npz")
        if os.path.exists(outp):
            skip += 1
            continue
        t0 = time.time()
        imgs = clip["imgs"]                                # [T,H,W,3] u8
        T = len(imgs)
        feats = np.empty((T, enc_dim), np.float16)
        for s in range(0, T, args.batch):
            chunk = torch.from_numpy(imgs[s:s + args.batch]).to(dev)
            chunk = chunk.permute(0, 3, 1, 2).float() / 255.0        # [b,3,H,W]
            fv = policy.encode_frames(chunk.unsqueeze(1))[:, 0]       # [b,enc_dim]
            feats[s:s + chunk.shape[0]] = fv.float().cpu().numpy().astype(np.float16)
        np.savez(outp, feats=feats, action=clip["action"], **clip["extra"])
        done += 1
        print(f"[encode] ✓ {clip['name']} T={T} {(time.time()-t0):.1f}s "
              f"[{done} done/{skip} skip]", flush=True)
    tot = sum(os.path.getsize(f) for f in glob.glob(os.path.join(args.out, "*.npz")))
    print(f"[encode] DONE {done} 新编/{skip} 跳过;{args.out} 共 {tot/1e9:.2f}GB", flush=True)


if __name__ == "__main__":
    main()
