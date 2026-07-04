# -*- coding: utf-8 -*-
"""R1:Context 塔(世界模型)预训练 —— fovea-twotower-step1 §4。

流 = 每帧 [81 DINO token + 1 动作 token] 因果交错,目标 = 下 token 潜变量 MSE。
DINOv2-S 冻结在线前向(126×126 → 9×9 patch);无缓存管理,分片即训。

用法(冒烟/正式只差 --steps/--bs):
    PYTHONPATH=. python train/fovea_twotower/train_r1.py \
        --data runs/data/g500_160p --out runs/ftt_r1 --steps 12000 --bs 8 --seq 64
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from net.fovea_twotower import ContextTower, act_featurize
from train.gaming500.dataset import Gaming500Dataset

IMN_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMN_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def dino_encode(dino, img_u8, bs=256):
    """[N,3,126,126] uint8 → [N,81,384] bf16(冻结,分块防峰值)。"""
    outs = []
    for i in range(0, img_u8.shape[0], bs):
        x = img_u8[i:i + bs].float().div_(255)
        x = (x - IMN_MEAN.to(x.device)) / IMN_STD.to(x.device)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            outs.append(dino.forward_features(x)["x_norm_patchtokens"])
    return torch.cat(outs).bfloat16()


def batch_to_stream(batch, dino, dev):
    img = batch["img"].to(dev, non_blocking=True)      # [B,L,3,S,S] u8
    B, L = img.shape[:2]
    lat = dino_encode(dino, img.flatten(0, 1)).view(B, L, 81, 384)
    act = act_featurize(*(batch[k].to(dev) for k in
                          ("dx", "dy", "keys", "gui", "dt"))).bfloat16()
    return lat, act


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="runs/data/g500_160p")
    p.add_argument("--out", default="runs/ftt_r1")
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--seq", type=int, default=64, help="窗口帧数(10Hz → 6.4s)")
    p.add_argument("--crop", default="resize", choices=["resize", "center", "random"],
                   help="resize=全图方形缩放; center=屏幕中心原生裁剪(凹区口径)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--holdout-frac", type=float, default=0.1)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"

    mk = lambda split: Gaming500Dataset(
        args.data, seq_len=args.seq, img_size=126, stride=args.seq,
        crop_mode=args.crop, split=split, holdout_frac=args.holdout_frac)
    dl = DataLoader(mk("train"), batch_size=args.bs, shuffle=True, drop_last=True,
                    num_workers=args.workers, pin_memory=True, persistent_workers=True)
    dl_ev = DataLoader(mk("holdout"), batch_size=args.bs, num_workers=2)

    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                          verbose=False).to(dev).eval()
    model = ContextTower().to(dev).bfloat16()
    n_par = sum(x.numel() for x in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(
        (s + 1) / args.warmup, 0.5 * (1 + torch.cos(torch.tensor(
            min(s / args.steps, 1.0) * 3.14159)).item())))
    logf = open(os.path.join(args.out, "log.jsonl"), "a")
    print(f"[R1] {n_par:.1f}M params | {len(dl.dataset)} train windows "
          f"| {len(dl_ev.dataset)} holdout windows", flush=True)

    step, t0, it = 0, time.time(), iter(dl)
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            continue
        lat, act = batch_to_stream(batch, dino, dev)
        loss = model(lat, act)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 50 == 0:
            rec = {"step": step, "loss": round(loss.item(), 5),
                   "gnorm": round(float(gn), 3),
                   "sps": round(step / (time.time() - t0), 3)}
            print(f"[R1] {rec}", flush=True)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            ev, n = 0.0, 0
            with torch.no_grad():
                for b in dl_ev:
                    lat, act = batch_to_stream(b, dino, dev)
                    ev += model(lat, act).item()
                    n += 1
                    if n >= 20:
                        break
            model.train()
            rec = {"step": step, "eval_loss": round(ev / max(n, 1), 5)}
            print(f"[R1] {rec}", flush=True)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            torch.save({"model": model.state_dict(), "step": step, "args": vars(args)},
                       os.path.join(args.out, "ckpt.pt"))
    print(f"[R1] done {step} steps, ckpt → {args.out}/ckpt.pt", flush=True)


if __name__ == "__main__":
    main()
