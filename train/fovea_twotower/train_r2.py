# -*- coding: utf-8 -*-
"""R2/B1:Action 塔流匹配 BC —— fovea-twotower-step1 §4 的核心对比。

同一脚本三种跑法,唯一差别是历史通道:
    --seed 1   R2 主案:GDN 状态播种自冻结 Context 塔;
    --seed 0   B1 消融:零状态(只见当前帧);
    --seed 2   B1.5 对照(选跑,R2>B1 后):错配播种——batch 内 roll 1,
               拿别人的历史状态播给你。若 R2(匹配)≈ R2(错配),播种收益只是
               状态统计先验而非本局历史内容,S3a 结论降级。
其余(同源初始化、数据、超参、评测)完全一致;--rng 固定全局随机种子,
各组间初始化/数据顺序/流匹配 ε 全配对,唯一自由变量 = 播种条件。

评测(holdout,4 步 Euler 采样,固定 generator 消除评测噪声方差):
    f1_attack(S3a 主指标)/ f1_keys(20 键宏平均)/ r2_mouse(对数域 dx,dy)。

用法:
    PYTHONPATH=. python train/fovea_twotower/train_r2.py \
        --data runs/data/g500_160p --ctx runs/ftt_r1/ckpt.pt \
        --out runs/ftt_r2_seed --seed 1 --steps 6000
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from net.fovea_twotower import ActionTower, ContextTower
from train.fovea_twotower.train_r1 import batch_to_stream
from train.gaming500.dataset import Gaming500Dataset

H = 8                                                  # chunk 长(0.8s@10Hz)


@torch.no_grad()
def context_states(ctx, lat, act, t, mode):
    """冻结塔吃 0..t 帧交错流 → (第 t 帧潜变量, GDN 状态|None)。

    mode: 0=零状态(B1) 1=匹配播种(R2) 2=错配播种(B1.5,batch 内 roll 1)。"""
    if mode == 0:
        return lat[:, t], None
    _, states = ctx.encode(lat[:, :t + 1], act[:, :t], want_states=True)
    if mode == 2:
        states = [{"recurrent_state": st["recurrent_state"].roll(1, dims=0),
                   "conv_state": tuple(c.roll(1, dims=0) for c in st["conv_state"])}
                  for st in states]
    return lat[:, t], states


def prep(batch, dino, ctx, mode, dev):
    lat, act = batch_to_stream(batch, dino, dev)
    L = lat.shape[1]
    t = L - 1 - H                                      # 留 H 个未来区间当目标
    lat_now, states = context_states(ctx, lat, act, t, mode)
    z1 = act[:, t:t + H]                               # [B,H,24] 目标 chunk
    return lat_now, states, z1


@torch.no_grad()
def evaluate(model, ctx, dino, dl, mode, dev, n_max=640):
    tp = fp = fn = 0
    tp_a = fp_a = fn_a = 0
    sse = sst = 0.0
    mice, n = [], 0
    gen = torch.Generator(dev).manual_seed(1234)       # 固定采样噪声,评测可比
    for batch in dl:
        lat_now, states, z1 = prep(batch, dino, ctx, mode, dev)
        pred = model.sample(lat_now, seed=states, steps=4, generator=gen)
        kp, kt = (pred[..., 2:22] > 0.5), (z1[..., 2:22] > 0.5)
        tp += (kp & kt).sum().item()
        fp += (kp & ~kt).sum().item()
        fn += (~kp & kt).sum().item()
        ap, at = kp[..., 7], kt[..., 7]                # key_attack 位
        tp_a += (ap & at).sum().item()
        fp_a += (ap & ~at).sum().item()
        fn_a += (~ap & at).sum().item()
        mp, mt = pred[..., :2].float(), z1[..., :2].float()
        mice.append(mt)
        sse += ((mp - mt) ** 2).sum().item()
        n += z1.shape[0]
        if n >= n_max:
            break
    mt_all = torch.cat(mice)
    sst = ((mt_all - mt_all.mean()) ** 2).sum().item()
    f1 = lambda tp, fp, fn: 2 * tp / max(2 * tp + fp + fn, 1)
    return {"f1_keys": round(f1(tp, fp, fn), 4),
            "f1_attack": round(f1(tp_a, fp_a, fn_a), 4),
            "r2_mouse": round(1 - sse / max(sst, 1e-9), 4), "n_eval": n}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="runs/data/g500_160p")
    p.add_argument("--ctx", default="runs/ftt_r1/ckpt.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, choices=[0, 1, 2], required=True,
                   help="1=播种(R2) 0=零状态(B1) 2=错配播种(B1.5)")
    p.add_argument("--rng", type=int, default=0,
                   help="全局随机种子;各条件用同一 --rng 即完全配对")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--crop", default=None, choices=[None, "resize", "center", "random"],
                   help="默认沿用 ctx ckpt 训练时的 crop 口径")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev, mode = "cuda", args.seed
    torch.manual_seed(args.rng)                        # 配对:init/数据顺序/ε 全同
    crop = args.crop or torch.load(args.ctx, map_location="cpu").get(
        "args", {}).get("crop", "resize")

    mk = lambda split, sh: DataLoader(
        Gaming500Dataset(args.data, seq_len=args.seq, img_size=126,
                         stride=args.seq // 2, crop_mode=crop,
                         split=split, holdout_frac=0.1),
        batch_size=args.bs, shuffle=sh, drop_last=sh, num_workers=args.workers,
        pin_memory=True, persistent_workers=True)
    dl, dl_ev = mk("train", True), mk("holdout", False)

    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                          verbose=False).to(dev).eval()
    ctx = ContextTower().to(dev).bfloat16().eval()
    ctx.load_state_dict(torch.load(args.ctx, map_location=dev)["model"])
    for q in ctx.parameters():
        q.requires_grad_(False)                        # 冻结(Table 2 配方)
    model = ActionTower(horizon=H).to(dev).bfloat16()
    model.init_from(ctx)                               # 同源初始化

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(
        (s + 1) / args.warmup, 0.5 * (1 + torch.cos(torch.tensor(
            min(s / args.steps, 1.0) * 3.14159)).item())))
    logf = open(os.path.join(args.out, "log.jsonl"), "a")
    print(f"[R2 seed={args.seed}] {len(dl.dataset)}/{len(dl_ev.dataset)} windows",
          flush=True)

    step, t0, it = 0, time.time(), iter(dl)
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            continue
        lat_now, states, z1 = prep(batch, dino, ctx, mode, dev)
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
            print(f"[R2 seed={args.seed}] {rec}", flush=True)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            rec = {"step": step, **evaluate(model, ctx, dino, dl_ev, mode, dev)}
            model.train()
            print(f"[R2 seed={args.seed}] EVAL {rec}", flush=True)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            torch.save({"model": model.state_dict(), "step": step,
                        "args": vars(args)}, os.path.join(args.out, "ckpt.pt"))
    print(f"[R2 seed={args.seed}] done → {args.out}/ckpt.pt", flush=True)


if __name__ == "__main__":
    main()
