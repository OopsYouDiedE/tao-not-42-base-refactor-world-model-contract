# -*- coding: utf-8 -*-
"""W1:Context 塔 + 周边消息通道 —— fovea-twotower-step2 §3。

流 = 每帧 [81 DINO token | 1 消息 token | 1 动作 token],目标 = 下帧潜变量 MSE
(不变,消息位不作目标):消息被保持的唯一梯度压力 = 周边动静迟早进凹区,
记住带时消息有预测价值(step2 §2 S4a 的一档判定,无循环论证)。

用法:
    PYTHONPATH=. python train/fovea_twotower/train_w1.py \
        --data runs/data/g500_360p --out runs/ftt_w1 --steps 6000 --bs 4
"""
import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from net.fovea_twotower import ContextTower
from train.fovea_twotower.train_r1 import batch_to_stream
from train.gaming500.dataset import Gaming500Dataset, N_MSG


def batch_to_stream_msg(batch, dino, dev):
    lat, act = batch_to_stream(batch, dino, dev)
    msg = batch["msg"].to(dev, non_blocking=True).bfloat16()
    return lat, act, msg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="runs/data/g500_360p")
    p.add_argument("--out", default="runs/ftt_w1")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--crop", default="center")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--aux-msg", type=float, default=0.0,
                   help="S4a 二档:消息位预测下帧消息的辅助损失权重(0=关)")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"

    mk = lambda split: Gaming500Dataset(
        args.data, seq_len=args.seq, img_size=126, stride=args.seq,
        crop_mode=args.crop, split=split, holdout_frac=0.1, periph=True)
    dl = DataLoader(mk("train"), batch_size=args.bs, shuffle=True, drop_last=True,
                    num_workers=args.workers, pin_memory=True, persistent_workers=True)
    dl_ev = DataLoader(mk("holdout"), batch_size=args.bs, num_workers=2)

    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                          verbose=False).to(dev).eval()
    model = ContextTower(n_msg=N_MSG, aux_msg=args.aux_msg).to(dev).bfloat16()
    n_par = sum(x.numel() for x in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(
        (s + 1) / args.warmup, 0.5 * (1 + torch.cos(torch.tensor(
            min(s / args.steps, 1.0) * 3.14159)).item())))
    logf = open(os.path.join(args.out, "log.jsonl"), "a")
    print(f"[W1] {n_par:.1f}M params | {len(dl.dataset)}/{len(dl_ev.dataset)} windows",
          flush=True)

    step, t0, ev_seen, ev_evt, it = 0, time.time(), 0, 0, iter(dl)
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            continue
        lat, act, msg = batch_to_stream_msg(batch, dino, dev)
        ev_seen += msg.shape[0] * msg.shape[1]
        ev_evt += int((msg[..., :8].sum(-1) > 0).sum())
        loss = model(lat, act, msg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 50 == 0:
            rec = {"step": step, "loss": round(loss.item(), 5),
                   "gnorm": round(float(gn), 3),
                   "evt_rate": round(ev_evt / max(ev_seen, 1), 4),
                   "sps": round(step / (time.time() - t0), 3)}
            print(f"[W1] {rec}", flush=True)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            ev, n = 0.0, 0
            with torch.no_grad():
                for b in dl_ev:
                    lat, act, msg = batch_to_stream_msg(b, dino, dev)
                    ev += model(lat, act, msg).item()
                    n += 1
                    if n >= 20:
                        break
            model.train()
            rec = {"step": step, "eval_loss": round(ev / max(n, 1), 5)}
            print(f"[W1] EVAL {rec}", flush=True)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            torch.save({"model": model.state_dict(), "step": step, "args": vars(args)},
                       os.path.join(args.out, "ckpt.pt"))
    print(f"[W1] done {step} steps → {args.out}/ckpt.pt", flush=True)


if __name__ == "__main__":
    main()
