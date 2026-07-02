#!/usr/bin/env python3
"""Minecraft VPT 离线 Dreamer4 世界模型训练(Dreamer 4 论文的离线预训练阶段)。

数据:train/minecraft/vpt_dataset.VPTStreamDataset(mp4+jsonl,frame_skip=1,64px)。
模型:net/dreamer4.WorldModel(连续潜 token tokenizer + 因果时空 Transformer +
shortcut forcing 速度头)。VPT 数据无奖励 ⇒ 只训 tokenizer 重建 + 流匹配/自一致,
reward/cont 头留给在线阶段(train/craftground/train_dreamer4)。

动作契约:VPT 连续 22 维动作向量(act_agg;鼠标 2 + 二值键 20),经 dynamics 的
action_proj 线性注入——Dreamer4 的 AdaLN 调制不要求 one-hot。

评估(holdout clip):
  - psnr_gen:     context → 4 步 Euler 流生成下一帧 token → 解码,vs 真值下一帧
  - psnr_recon:   tokenizer 重建上限
  - psnr_persist: 持续性基线(上一帧当预测)。psnr_gen 必须超过它才说明动力学
    学到了"动作驱动的变化",而不是在复读上一帧。

使用方法:
    python -m train.minecraft.train_dreamer4 --data_dir runs/data/vpt_findcave \
        --camera_scale 29 --total_steps 4000
"""
import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from net.dreamer4 import Dreamer4Config, WorldModel
from train.minecraft.vpt_action import ACTION_DIM
from train.minecraft.vpt_dataset import VPTStreamDataset


def parse_args():
    p = argparse.ArgumentParser(description="Minecraft VPT 离线 Dreamer4 世界模型训练")
    p.add_argument("--data_dir", default="runs/data/vpt_findcave")
    p.add_argument("--holdout_n", type=int, default=3)
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=16)
    p.add_argument("--camera_scale", type=float, default=29.0)
    p.add_argument("--frame_skip", type=int, default=1,
                   help="可变时间跨度上限:每个转移 Δt~U{1..frame_skip}(帧)。>1 时 jumpy 预测,"
                        "persistence 基线随运动量增大而变弱,动作效应信噪比更高(数学动机见 "
                        "vpt_dataset docstring);Δt 经 DT_NORM 归一化后追加为条件向量末维")
    p.add_argument("--token_dim", type=int, default=256)
    p.add_argument("--dyn_layers", type=int, default=4)
    p.add_argument("--dyn_heads", type=int, default=8)
    p.add_argument("--enc_base", type=int, default=32,
                   help="tokenizer 编码器基础通道(各级 = b,2b,4b,8b;解码器倒序)")
    p.add_argument("--shortcut_hidden", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--total_steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--sc_weight", type=float, default=1.0)
    p.add_argument("--d_min", type=float, default=0.125)
    p.add_argument("--gen_steps", type=int, default=4)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--clip_cache", type=int, default=4)
    p.add_argument("--amp", choices=["off", "bf16", "fp16"], default="bf16",
                   help="混合精度:autocast 前向/反向(bf16 无需 GradScaler,fp16 需要;"
                        "评估始终 fp32 保证指标口径)。危险算子(norm/softmax/loss)由 "
                        "autocast 自动保持 fp32(I4)")
    p.add_argument("--eval_interval", type=int, default=400)
    p.add_argument("--n_eval_batches", type=int, default=8)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--run_dir", default="runs/minecraft_d4_offline")
    p.add_argument("--resume", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# Δt 条件的固定归一化尺度(帧)。离线/在线共用:在线 20Hz 单步 Δt=1 ⇒ 特征 0.25。
# 是契约常数,不随 --frame_skip 变(否则不同 frame_skip 训练的 ckpt 无法互迁)。
DT_NORM = 4.0


def make_batch(sample, device):
    """dataset batch → (image [B,T,3,H,W] float01, cond [B,T,A+1])。

    act_agg[t] 是转移 t→t+1 的动作,只有 T-1 个;首位补零对齐到 T 帧
    (dynamics 因果时间注意下,context[t] 消费的是"进入第 t+1 帧前"的动作序列)。
    条件向量末维 = Δt/DT_NORM(jumpy 预测时模型必须知道要预测多远)。
    """
    img = sample["img"].to(device, non_blocking=True).float() / 255.0
    act = sample["act_agg"].to(device, non_blocking=True)          # [B,T-1,A]
    dt = sample["dt"].to(device, non_blocking=True)                # [B,T-1]
    cond = torch.zeros(img.shape[0], img.shape[1], act.shape[-1] + 1, device=device)
    cond[:, 1:, :-1] = act
    cond[:, 1:, -1] = dt / DT_NORM
    return img, cond


@torch.no_grad()
def evaluate(wm, hold_iter, n_batches, device, gen_steps):
    """holdout:生成/重建/持续性 PSNR 与流匹配验证损失。"""
    wm.eval()
    agg = {"psnr_gen": 0.0, "psnr_recon": 0.0, "psnr_persist": 0.0, "val_flow": 0.0}
    for _ in range(n_batches):
        img, act = make_batch(next(hold_iter), device)
        m = wm.eval_next_frame(img, act, gen_steps=gen_steps)
        _, tm = wm.loss(img, act)
        for k in ("psnr_gen", "psnr_recon", "psnr_persist"):
            agg[k] += m[k] / n_batches
        agg["val_flow"] += tm["flow"] / n_batches
    wm.train()
    return agg


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True     # 见 knowledge/dreamer.md §2.5
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    print("=" * 78, flush=True)
    print("🌍 Minecraft VPT 离线 Dreamer4 世界模型")
    print(f"   data={args.data_dir} | img={args.img_size} | seq={args.seq_len} "
          f"| token_dim={args.token_dim} | dyn_layers={args.dyn_layers}")
    print("=" * 78, flush=True)

    ds_kw = dict(seq_len=args.seq_len, img_size=args.img_size,
                 camera_scale=args.camera_scale, frame_skip=args.frame_skip,
                 clip_cache=args.clip_cache, seed=args.seed)
    train_ds = VPTStreamDataset(args.data_dir, split="train",
                                holdout_n=args.holdout_n, **ds_kw)
    hold_ds = VPTStreamDataset(args.data_dir, split="holdout",
                               holdout_n=args.holdout_n, **ds_kw)
    train_iter = iter(DataLoader(train_ds, batch_size=args.batch_size,
                                 num_workers=args.workers, pin_memory=True))
    hold_iter = iter(DataLoader(hold_ds, batch_size=args.batch_size,
                                num_workers=1, pin_memory=True))

    b_ = args.enc_base
    cfg = Dreamer4Config(
        obs_shape=(3, args.img_size, args.img_size), num_actions=ACTION_DIM + 1,
        token_dim=args.token_dim, dyn_layers=args.dyn_layers, dyn_heads=args.dyn_heads,
        enc_depths=(b_, 2 * b_, 4 * b_, 8 * b_),
        dec_depths=(8 * b_, 4 * b_, 2 * b_, b_),
        shortcut_hidden=args.shortcut_hidden,
    )
    wm = WorldModel(cfg).to(device)
    n_params = sum(p.numel() for p in wm.parameters())
    print(f"✅ Dreamer4 世界模型: {n_params/1e6:.2f}M 参数 "
          f"(token 网格 {wm.tokenizer.grid}, S={wm.num_tokens})", flush=True)

    optimizer = torch.optim.AdamW(wm.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / max(args.warmup_steps, 1)))

    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        wm.load_state_dict(ck["wm"])
        optimizer.load_state_dict(ck["optimizer"])
        start_step = ck.get("step", 0)
        print(f"♻️  已恢复 {args.resume}(step={start_step})", flush=True)

    os.makedirs(args.run_dir, exist_ok=True)

    def save_ckpt(tag, step, metrics=None):
        path = os.path.join(args.run_dir, f"{tag}.pt")
        torch.save({"wm": wm.state_dict(), "optimizer": optimizer.state_dict(),
                    "step": step, "cfg": vars(args), "metrics": metrics}, path)
        print(f"💾 已保存 {path}", flush=True)

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)
    use_amp = amp_dtype is not None and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and args.amp == "fp16")
    print(f"⚙️  混合精度: {args.amp if use_amp else 'off(fp32)'}", flush=True)

    best_gen = -float("inf")
    t0 = time.time()
    wm.train()
    try:
        for step in range(start_step, args.total_steps):
            img, act = make_batch(next(train_iter), device)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                total, m = wm.loss(img, act, d_min=args.d_min, sc_weight=args.sc_weight)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(wm.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            sched.step()

            if step % args.log_interval == 0:
                fps = args.batch_size * args.seq_len * (step - start_step + 1) \
                    / max(time.time() - t0, 1e-6)
                print(f"[{step:5d}/{args.total_steps}] total={total.item():.4f} "
                      f"recon={m['recon']:.4f} flow={m['flow']:.4f} sc={m['sc']:.4f} "
                      f"| {fps:.0f} 帧/s", flush=True)

            if (step + 1) % args.eval_interval == 0 or step + 1 == args.total_steps:
                e = evaluate(wm, hold_iter, args.n_eval_batches, device, args.gen_steps)
                print(f"    📊 holdout@{step+1}: gen={e['psnr_gen']:.2f}dB "
                      f"recon={e['psnr_recon']:.2f}dB persist={e['psnr_persist']:.2f}dB "
                      f"val_flow={e['val_flow']:.4f}", flush=True)
                if e["psnr_gen"] > best_gen:
                    best_gen = e["psnr_gen"]
                    save_ckpt("best", step + 1, e)
    except KeyboardInterrupt:
        print("\n⏹️  训练中断", flush=True)
    finally:
        save_ckpt("final", args.total_steps)

    print(f"✅ 完成,耗时 {(time.time()-t0)/60:.1f} 分钟,best psnr_gen={best_gen:.2f}dB",
          flush=True)


if __name__ == "__main__":
    main()
