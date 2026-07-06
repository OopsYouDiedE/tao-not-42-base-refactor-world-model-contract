#!/usr/bin/env python3
"""预编译 gaming500-MC HDF5 分片 → 冻结骨干 CLS 特征,存快盘,供快头 BC 免解码/免 DINO 前向直训。

encode_vpt_feats.py 的 gaming500 版:输入不是 mp4+jsonl 而是我们编码档 HDF5
(tests/encode_gaming500_hdf5.py 产出,f[game][name] 分组,jpeg/dx/dy/keys/gui/frame_idx)。
产物 npz 契约与 VPT 版**逐字段一致**(训练侧无需分叉):
  feats  fp16 [T, 384]   逐帧 CLS(BCPolicy.encode_frames,同一 backbone+预处理)
  action fp32 [T, 22]    [dx/scale, dy/scale, keys×20](vpt_dataset._action_vec 契约;
                         KEY_NAMES ≡ VPT_KEYS 逐位相同,位掩码直接展开)
  task   str             会话 metadata 标题
额外字段(VPT 版没有,向后兼容):
  dx_px/dy_px fp32 [T]   原始像素位移(未归一)——相机尺度是训练期超参,改 scale 免重编码
  gui        uint8 [T]   区间内是否开 GUI(max 聚合)

动作对齐:h5 的 dx/dy/keys/gui 按 30Hz 源事件帧存,图像 15Hz;action[j] = 图 j→j+1
区间 (fidx[j], fidx[j+1]] 的聚合(dx/dy 求和、keys 按位 OR、gui 取 max),与
train/gaming500/dataset.Gaming500Dataset.__getitem__ 同一语义;末帧动作置 0。
⚠ 15Hz 聚合使 |dx| 分布约为 30Hz 逐帧的 2×,CAMERA_SCALE=10 截断更多——脚本逐段打印
dx p50/p99 供校准;归一 action 与原始 dx_px 双存,训练侧可自行换档。

分盘(遵用户规则):HDF5 留数据盘 /data;feats 存 nvme 快盘(默认 runs/data/g500_mc_feat)。
体积:CLS 0.75KB/帧,15 会话 ~35 万帧 ≈ 仅数百 MB。增量:已存在同名 .npz 跳过。

用法(需 GPU,排在 W4c 评测之后;dinov3 首次拉权重需 HF token + SOCKS):
  export HF_TOKEN=... ALL_PROXY=socks5://127.0.0.1:2080 HTTPS_PROXY=socks5://127.0.0.1:2080
  PYTHONPATH=. python tests/encode_g500_feats.py \
      --data /data/gaming500_mc_h5 --out runs/data/g500_mc_feat --img-size 128
"""
import argparse
import glob
import json
import os
import time

import cv2
import h5py
import numpy as np
import torch

from net.bc import build_bc_policy, BCConfig
from net.config import BackboneConfig
from train.gaming500.dataset import unpack_keys, KEY_NAMES
from train.minecraft.vpt_action import CAMERA_SCALE
from train.minecraft.vpt_dataset import VPT_KEYS

assert KEY_NAMES == VPT_KEYS, "gaming500 KEY_NAMES 与 VPT_KEYS 位序不一致,动作契约失效"


def iter_parts(data_dir):
    """遍历所有分片的所有 (part_name, shard_path);part 名全局唯一(session_序号)。"""
    for shp in sorted(glob.glob(os.path.join(data_dir, "shard_*.h5"))):
        with h5py.File(shp, "r") as f:
            for game in f:
                for name in f[game]:
                    yield f"{game}/{name}", shp


def part_actions(g, camera_scale):
    """图像帧区间聚合动作 → (action [T,22], dx_px, dy_px, gui)。语义同 Gaming500Dataset。"""
    fidx = g["frame_idx"][:].astype(np.int64)
    dx_f, dy_f = g["dx"][:], g["dy"][:]
    keys_f, gui_f = g["keys"][:], g["gui"][:]
    T, M = len(fidx), len(dx_f)
    dx = np.zeros(T, np.float32)
    dy = np.zeros(T, np.float32)
    keys = np.zeros((T, len(VPT_KEYS)), np.uint8)
    gui = np.zeros(T, np.uint8)
    for j in range(T - 1):
        lo, hi = min(fidx[j] + 1, M), min(fidx[j + 1] + 1, M)
        if lo >= hi:
            continue
        dx[j] = dx_f[lo:hi].sum()
        dy[j] = dy_f[lo:hi].sum()
        keys[j] = np.bitwise_or.reduce(unpack_keys(keys_f[lo:hi]), axis=0)
        gui[j] = gui_f[lo:hi].max()
    act = np.zeros((T, 2 + len(VPT_KEYS)), np.float32)
    act[:, 0] = np.clip(dx / camera_scale, -1.0, 1.0)
    act[:, 1] = np.clip(dy / camera_scale, -1.0, 1.0)
    act[:, 2:] = keys
    return act, dx, dy, gui


def decode_jpegs(jpeg_ds, img_size):
    """JPEG object 列 → uint8 [T,H,W,3] RGB(resize 到 img_size)。"""
    out = np.empty((len(jpeg_ds), img_size, img_size, 3), np.uint8)
    for i, buf in enumerate(jpeg_ds):
        im = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        out[i] = cv2.resize(im, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return out


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/data/gaming500_mc_h5", help="编码档 HDF5 分片目录(数据盘)")
    p.add_argument("--out", default="runs/data/g500_mc_feat", help="feats 输出(快盘 nvme)")
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--backbone", choices=["dinov3", "dinov2"], default="dinov3")
    p.add_argument("--camera-scale", type=float, default=CAMERA_SCALE)
    p.add_argument("--batch", type=int, default=256, help="DINO 前向的帧批大小")
    p.add_argument("--limit", type=int, default=0, help=">0 只编前 N 段(冒烟)")
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    policy = build_bc_policy(BCConfig(backbone=BackboneConfig(kind=args.backbone))).to(dev).eval()
    enc_dim = policy.enc_dim

    manifest = {"backbone": args.backbone, "img_size": args.img_size,
                "camera_scale": args.camera_scale, "enc_dim": enc_dim,
                "action_keys": VPT_KEYS, "source": os.path.abspath(args.data)}
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    parts = list(iter_parts(args.data))
    if args.limit:
        parts = parts[:args.limit]
    print(f"[encode] {len(parts)} parts in {args.data} | backbone={args.backbone} "
          f"img={args.img_size} enc_dim={enc_dim} scale={args.camera_scale} → {args.out}",
          flush=True)
    done = skip = 0
    for key, shp in parts:
        name = key.split("/")[-1]
        outp = os.path.join(args.out, name + ".npz")
        if os.path.exists(outp):
            skip += 1
            continue
        t0 = time.time()
        with h5py.File(shp, "r") as f:
            g = f[key]
            task = json.loads(g.attrs.get("meta_json", "{}")).get("title", "") \
                or str(g.attrs.get("task", ""))
            act, dx, dy, gui = part_actions(g, args.camera_scale)
            imgs = decode_jpegs(g["jpeg"], args.img_size)
        T = len(imgs)
        feats = np.empty((T, enc_dim), np.float16)
        for s in range(0, T, args.batch):
            chunk = torch.from_numpy(imgs[s:s + args.batch]).to(dev)
            chunk = chunk.permute(0, 3, 1, 2).float() / 255.0          # [b,3,H,W]
            fv = policy.encode_frames(chunk.unsqueeze(1))[:, 0]         # [b,enc_dim]
            feats[s:s + chunk.shape[0]] = fv.float().cpu().numpy().astype(np.float16)
        np.savez(outp, feats=feats, action=act, dx_px=dx, dy_px=dy, gui=gui, task=task)
        done += 1
        adx = np.abs(dx[:-1])
        print(f"[encode] ✓ {name} T={T} {(time.time()-t0):.1f}s "
              f"|dx| p50={np.percentile(adx,50):.1f} p99={np.percentile(adx,99):.1f} "
              f"clip%={100*np.mean(np.abs(act[:,0])>=1.0):.1f} "
              f"[{done} done/{skip} skip]", flush=True)
    tot = sum(os.path.getsize(f) for f in glob.glob(os.path.join(args.out, "*.npz")))
    print(f"[encode] DONE {done} 新编/{skip} 跳过;{args.out} 共 {tot/1e9:.2f}GB", flush=True)


if __name__ == "__main__":
    main()
