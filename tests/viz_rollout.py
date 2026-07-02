#!/usr/bin/env python3
"""脑内推演可视化:holdout 上 K 步开环 rollout,导出三联对比视频合集与胶片图。

对外接口:命令行脚本(main);无库接口。

每帧画面 = [真值 GT | 脑内推演 DREAM | 复读 PERSIST] 三联(最近邻放大),顶栏标注
样本号/阶段/步数/两侧 PSNR。每个样本先播 context 段(三联同为真值,标 CONTEXT),
再播 horizon 步开环生成(模型只回喂自己生成的 token,不再看真值)。另为每个样本导出
一张胶片图(3 行 × horizon 列):一眼看清"复读在高运动步崩掉、模型跟不跟得上"。

视频用 ffmpeg libx264 编码(浏览器可播);cv2 的 mp4v 在 Colab 预览常不可播,不用。

使用方法:
    PYTHONPATH=. python tests/viz_rollout.py --ckpt runs/mc_d4_b128/best.pt \
        --data_dir runs/data/vpt_findcave --out runs/viz/b128
"""
import argparse
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from net.dreamer4 import Dreamer4Config, WorldModel
from train.minecraft.train_dreamer4 import make_batch
from train.minecraft.vpt_action import ACTION_DIM
from train.minecraft.vpt_dataset import VPTStreamDataset


def parse_args():
    p = argparse.ArgumentParser(description="开环 rollout 可视化")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", default="runs/data/vpt_findcave")
    p.add_argument("--out", default="runs/viz/rollout", help="输出前缀(生成 <out>.mp4 与 <out>_sNN.png)")
    p.add_argument("--n_samples", type=int, default=6)
    p.add_argument("--context", type=int, default=8)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--gen_steps", type=int, default=4)
    p.add_argument("--fps", type=float, default=3.0, help="播放帧率(慢放便于逐步看)")
    p.add_argument("--upscale", type=int, default=2, help="最近邻放大倍数")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_model(ckpt, device):
    """从 ckpt 里存的训练 args 重建同构模型(与 train_dreamer4 的构造逻辑一致)。"""
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    c = ck["cfg"]
    b_ = c["enc_base"]
    cfg = Dreamer4Config(
        obs_shape=(3, c["img_size"], c["img_size"]), num_actions=ACTION_DIM + 1,
        token_dim=c["token_dim"], dyn_layers=c["dyn_layers"], dyn_heads=c["dyn_heads"],
        enc_depths=(b_, 2 * b_, 4 * b_, 8 * b_), dec_depths=(8 * b_, 4 * b_, 2 * b_, b_),
        shortcut_hidden=c["shortcut_hidden"], dec_min_res=c["img_size"] // 16,
    )
    wm = WorldModel(cfg).to(device)
    wm.load_state_dict(ck["wm"])
    wm.eval()
    print(f"✅ {ckpt} @step {ck.get('step')} | img={c['img_size']} "
          f"S={wm.num_tokens} | {sum(p.numel() for p in wm.parameters())/1e6:.0f}M")
    return wm, c, ck.get("step")


def to_u8(img_chw):
    """[3,H,W] float01 → BGR uint8。"""
    x = (img_chw.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def psnr(a, b):
    mse = float(((a.clamp(0, 1) - b) ** 2).mean())
    return -10.0 * np.log10(max(mse, 1e-10))


def panel_frame(gt, dream, persist, up, header):
    """三联拼接 + 顶栏文字 → BGR uint8 大图。"""
    tiles = []
    for im, tag in ((gt, "GT"), (dream, "DREAM"), (persist, "PERSIST")):
        t = cv2.resize(to_u8(im), None, fx=up, fy=up, interpolation=cv2.INTER_NEAREST)
        cv2.putText(t, tag, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(t, tag, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        tiles.append(t)
    row = np.concatenate(tiles, axis=1)
    bar = np.full((28, row.shape[1], 3), 32, np.uint8)
    cv2.putText(bar, header, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
    return np.concatenate([bar, row], axis=0)


@torch.no_grad()
def rollout_frames(wm, img, act, context, horizon, gen_steps):
    """开环推演:返回逐步解码帧 dream[B,K,3,H,W](模型只回喂自己生成的 token)。"""
    tokens, _ = wm.tokenizer.encode(img)
    toks = tokens[:, :context]
    outs = []
    for k in range(horizon):
        ctx = wm.dynamics(toks, act[:, :context + k])
        nxt = wm.generate_next(ctx[:, -1:], steps=gen_steps)
        toks = torch.cat([toks, nxt], dim=1)
        outs.append((wm.tokenizer.decode(nxt)[:, 0] + 0.5).clamp(0, 1))
    return torch.stack(outs, dim=1)


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    wm, c, step = build_model(args.ckpt, device)

    T = args.context + args.horizon
    ds = VPTStreamDataset(args.data_dir, split="holdout", holdout_n=c["holdout_n"],
                          seq_len=T, img_size=c["img_size"], seed=args.seed,
                          camera_scale=c["camera_scale"], frame_skip=c.get("frame_skip", 1))
    img, act = make_batch(next(iter(DataLoader(ds, batch_size=args.n_samples,
                                               num_workers=0))), device)
    dream = rollout_frames(wm, img, act, args.context, args.horizon, args.gen_steps)
    last_real = img[:, args.context - 1]

    tmp = tempfile.mkdtemp(prefix="viz_rollout_")
    fi = 0
    for b in range(img.shape[0]):
        # context 段:三联同为真值
        for t in range(args.context):
            f = panel_frame(img[b, t], img[b, t], img[b, t], args.upscale,
                            f"sample {b}  CONTEXT {t + 1}/{args.context}  (step {step})")
            cv2.imwrite(os.path.join(tmp, f"{fi:05d}.png"), f)
            fi += 1
        # rollout 段:DREAM=开环生成,PERSIST=复读最后一帧真值
        strip = []
        for k in range(args.horizon):
            gt = img[b, args.context + k]
            pd, pp = psnr(dream[b, k], gt), psnr(last_real[b], gt)
            f = panel_frame(gt, dream[b, k], last_real[b], args.upscale,
                            f"sample {b}  ROLLOUT {k + 1}/{args.horizon}  "
                            f"dream {pd:.1f}dB vs persist {pp:.1f}dB")
            cv2.imwrite(os.path.join(tmp, f"{fi:05d}.png"), f)
            fi += 1
            strip.append(np.concatenate(
                [to_u8(gt), to_u8(dream[b, k]), to_u8(last_real[b])], axis=0))
        cv2.imwrite(f"{args.out}_s{b:02d}.png", np.concatenate(strip, axis=1))

    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                    "-i", os.path.join(tmp, "%05d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "20", f"{args.out}.mp4"], check=True)
    shutil.rmtree(tmp)
    print(f"🎬 视频: {args.out}.mp4 | 胶片图: {args.out}_sNN.png "
          f"(胶片行序: 上=GT 中=DREAM 下=PERSIST,列=推演步)")


if __name__ == "__main__":
    main()
