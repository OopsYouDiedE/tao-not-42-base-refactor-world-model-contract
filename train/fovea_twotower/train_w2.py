# -*- coding: utf-8 -*-
"""W2(M1/M0/Mscr):Action 塔流匹配 BC,消息通道三消融 —— step2 §3。

三种跑法共用同一冻结 W1 塔(不重训世界塔,只改状态计算时喂的消息):
    --mode 1   M1  真实消息流(主案)
    --mode 0   M0  消息内容置零(valid 位保留)——同塔同参,只删周边信息
    --mode 2   Mscr 消息窗口内时序打乱——内容边际保留,时间对齐破坏(S4c)
播种恒开(本实验隔离变量=消息内容/时序,非播种本身);消息不直通动作塔,
唯一到达路径 = 冻结塔状态 → 播种(检验"慢通道→Mamba 记忆→快通道使用"全链)。
打乱用独立 CPU Generator(seed 777),不动全局 RNG——各组 init/数据序/ε 保持配对。

用法:
    PYTHONPATH=. python train/fovea_twotower/train_w2.py \
        --ctx runs/ftt_w1/ckpt.pt --out runs/ftt_w2_m1 --mode 1 --rng 0
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from net.fovea_twotower import ActionTower, ContextTower
from train.fovea_twotower.train_w1 import batch_to_stream_msg
from train.gaming500.dataset import Gaming500Dataset, N_MSG

H = 8


def msg_variant(msg, mode, gen):
    """mode: 1=原样 0=内容置零(留 valid/pan) 2=窗口内时序打乱(帧0留原位)。"""
    if mode == 1:
        return msg
    if mode == 0:
        z = torch.zeros_like(msg)
        z[..., 10] = msg[..., 10]                      # valid 位保留,token 数不变
        return z
    B, L = msg.shape[:2]
    out = msg.clone()
    for i in range(B):
        perm = torch.randperm(L - 1, generator=gen) + 1
        out[i, 1:] = msg[i, perm]
    return out


@torch.no_grad()
def context_states(ctx, lat, act, msg, t, mode, gen):
    m = msg_variant(msg[:, :t + 1], mode, gen)
    _, states = ctx.encode(lat[:, :t + 1], act[:, :t], m, want_states=True)
    return lat[:, t], states


def prep(batch, dino, ctx, mode, gen, dev):
    lat, act, msg = batch_to_stream_msg(batch, dino, dev)
    L = lat.shape[1]
    t = L - 1 - H
    lat_now, states = context_states(ctx, lat, act, msg, t, mode, gen)
    return lat_now, states, act[:, t:t + H], msg[:, :t + 1]


@torch.no_grad()
def evaluate(model, ctx, dino, dl, mode, dev, n_max=640):
    tp = fp = fn = 0
    tp_a = fp_a = fn_a = 0
    n = 0
    gen = torch.Generator(dev).manual_seed(1234)
    sgen = torch.Generator().manual_seed(777)          # 打乱种子固定,评测可复现
    for batch in dl:
        lat_now, states, z1, _ = prep(batch, dino, ctx, mode, sgen, dev)
        pred = model.sample(lat_now, seed=states, steps=4, generator=gen)
        kp, kt = (pred[..., 2:22] > 0.5), (z1[..., 2:22] > 0.5)
        tp += (kp & kt).sum().item()
        fp += (kp & ~kt).sum().item()
        fn += (~kp & kt).sum().item()
        ap, at = kp[..., 7], kt[..., 7]
        tp_a += (ap & at).sum().item()
        fp_a += (ap & ~at).sum().item()
        fn_a += (~ap & at).sum().item()
        n += z1.shape[0]
        if n >= n_max:
            break
    f1 = lambda tp, fp, fn: 2 * tp / max(2 * tp + fp + fn, 1)
    return {"f1_keys": round(f1(tp, fp, fn), 4),
            "f1_attack": round(f1(tp_a, fp_a, fn_a), 4), "n_eval": n}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="runs/data/g500_360p")
    p.add_argument("--ctx", default="runs/ftt_w1/ckpt.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--mode", type=int, choices=[0, 1, 2], required=True)
    p.add_argument("--rng", type=int, default=0)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--workers", type=int, default=10)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev, mode = "cuda", args.mode
    torch.manual_seed(args.rng)
    sgen = torch.Generator().manual_seed(777)          # 打乱专用,独立于全局 RNG

    ck = torch.load(args.ctx, map_location=dev)
    crop = ck.get("args", {}).get("crop", "center")
    mk = lambda split, sh: DataLoader(
        Gaming500Dataset(args.data, seq_len=args.seq, img_size=126,
                         stride=args.seq // 2, crop_mode=crop, periph=True,
                         split=split, holdout_frac=0.1),
        batch_size=args.bs, shuffle=sh, drop_last=sh, num_workers=args.workers,
        pin_memory=True, persistent_workers=True)
    dl, dl_ev = mk("train", True), mk("holdout", False)

    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                          verbose=False).to(dev).eval()
    ctx = ContextTower(n_msg=N_MSG).to(dev).bfloat16().eval()
    ctx.load_state_dict(ck["model"])
    for q in ctx.parameters():
        q.requires_grad_(False)
    model = ActionTower(horizon=H).to(dev).bfloat16()
    model.init_from(ctx)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(
        (s + 1) / args.warmup, 0.5 * (1 + torch.cos(torch.tensor(
            min(s / args.steps, 1.0) * 3.14159)).item())))
    logf = open(os.path.join(args.out, "log.jsonl"), "a")
    print(f"[W2 mode={mode}] {len(dl.dataset)}/{len(dl_ev.dataset)} windows", flush=True)

    step, t0, it = 0, time.time(), iter(dl)
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            continue
        lat_now, states, z1, _ = prep(batch, dino, ctx, mode, sgen, dev)
        B = z1.shape[0]
        eps = torch.randn_like(z1)
        tau = torch.rand(B, device=dev, dtype=z1.dtype)
        x_tau = (1 - tau[:, None, None]) * eps + tau[:, None, None] * z1
        v = model(lat_now, x_tau, tau, seed=states)
        loss = F.mse_loss(v, (z1 - eps))
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
            print(f"[W2 mode={mode}] {rec}", flush=True)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            rec = {"step": step, **evaluate(model, ctx, dino, dl_ev, mode, dev)}
            model.train()
            print(f"[W2 mode={mode}] EVAL {rec}", flush=True)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            torch.save({"model": model.state_dict(), "step": step,
                        "args": vars(args)}, os.path.join(args.out, "ckpt.pt"))
    print(f"[W2 mode={mode}] done → {args.out}/ckpt.pt", flush=True)


if __name__ == "__main__":
    main()
