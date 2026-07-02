#!/usr/bin/env python3
"""Minecraft VPT 离线行为克隆(BC)训练循环。

数据契约:train/minecraft/vpt_dataset.VPTStreamDataset(mp4+jsonl 成对裸数据,
frame_skip=1 逐帧转移)。模型:net/bc(冻结 DINO 骨干 + 因果时序 Transformer)。

监督目标(a_t | o_{≤t}, a_{<t}):
  - 相机 dx/dy → mu-law 分箱 CE(camera_to_bin;MSE 下"恒预测 0"是平凡解)
  - 20 个二值键 → BCE

评估(holdout 切分,与训练不同 clip):
  - 相机 bin top-1 准确率 vs 多数 bin 基线 / 持续性基线(抄上一步动作)
  - 按键 micro-F1 vs 全零基线(F1=0)/ 持续性基线
  持续性基线是关键对照:键位是长按型信号,任何低于它的模型没学到视觉信息。

使用方法:
    python -m train.minecraft.train_bc --data_dir runs/data/vpt_findcave \
        --camera_scale 20 --total_steps 3000
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from net.bc import BCConfig, build_bc_policy
from net.config import BackboneConfig
from train.minecraft.vpt_action import ACTION_DIM, CAMERA_BINS, N_MOUSE, camera_to_bin
from train.minecraft.vpt_dataset import VPTStreamDataset


def parse_args():
    p = argparse.ArgumentParser(description="Minecraft VPT 离线行为克隆训练")
    p.add_argument("--data_dir", default="runs/data/vpt_findcave")
    p.add_argument("--holdout_dir", default=None,
                   help="独立 holdout 目录;缺省用同目录按文件名切末 holdout_n 个 clip")
    p.add_argument("--holdout_n", type=int, default=2)
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--seq_len", type=int, default=16)
    p.add_argument("--camera_scale", type=float, default=20.0,
                   help="相机归一化尺度(按真数据 |dx| 分布校准,见 tests/download_vpt_data.py 报告)")
    p.add_argument("--backbone", choices=["dinov3", "dinov2"], default="dinov3")
    p.add_argument("--d", type=int, default=384)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--total_steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--key_loss_coeff", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--clip_cache", type=int, default=3)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--n_eval_batches", type=int, default=8)
    p.add_argument("--log_interval", type=int, default=25)
    p.add_argument("--run_dir", default="runs/minecraft_bc_v1")
    p.add_argument("--resume", default=None, help="从 checkpoint 恢复(仅可训练权重)")
    p.add_argument("--eval_only", action="store_true", default=False)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                   help="bf16 autocast(骨干+策略前向);L4/A100 原生支持,无需 GradScaler")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def make_batch(sample, device):
    """dataset batch → (img [B,T',3,H,W] float01, prev_act, target_act)。

    时间对齐:act_agg[t] 是转移 t→t+1 的动作。策略在观测 o_t(与 a_{<t})下预测 a_t,
    因此输入帧取 0..T-2(共 T'=T-1 帧),目标 act_agg[0..T-2],上一步动作右移一位补零。
    """
    img = sample["img"][:, :-1].to(device, non_blocking=True).float() / 255.0
    target = sample["act_agg"].to(device, non_blocking=True)          # [B,T',A]
    prev = torch.zeros_like(target)
    prev[:, 1:] = target[:, :-1]
    return img, prev, target


def bc_losses(cam_logits, key_logits, target):
    """CE(相机两轴 mu-law bin) + BCE(二值键)。返回 (cam_ce, key_bce, cam_tgt_bins)。"""
    cam_tgt = camera_to_bin(target[..., :N_MOUSE])                    # [B,T',2] long
    cam_ce = F.cross_entropy(
        cam_logits.flatten(0, 2).float(), cam_tgt.flatten())
    key_bce = F.binary_cross_entropy_with_logits(
        key_logits.float(), target[..., N_MOUSE:])
    return cam_ce, key_bce, cam_tgt


@torch.no_grad()
def evaluate(policy, holdout_iter, n_batches, device, use_amp):
    """holdout 指标:相机 bin 准确率(+多数 bin/持续性基线)与按键 micro-F1(+持续性基线)。"""
    policy.eval()
    cam_correct = cam_total = 0
    cam_persist_correct = 0
    key_tp = key_fp = key_fn = 0
    p_tp = p_fp = p_fn = 0
    cam_ce_sum = key_bce_sum = 0.0
    all_cam_bins = []
    for _ in range(n_batches):
        sample = next(holdout_iter)
        img, prev, target = make_batch(sample, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            feats = policy.encode_frames(img)
            cam_logits, key_logits = policy(feats.float(), prev)
        cam_ce, key_bce, cam_tgt = bc_losses(cam_logits, key_logits, target)
        cam_ce_sum += cam_ce.item()
        key_bce_sum += key_bce.item()

        pred_bins = cam_logits.float().argmax(-1)                     # [B,T',2]
        cam_correct += (pred_bins == cam_tgt).sum().item()
        cam_total += cam_tgt.numel()
        all_cam_bins.append(cam_tgt.flatten().cpu())
        # 持续性基线:抄上一步动作的相机 bin / 键位
        prev_bins = camera_to_bin(prev[..., :N_MOUSE])
        cam_persist_correct += (prev_bins == cam_tgt).sum().item()

        key_tgt = target[..., N_MOUSE:] > 0.5
        key_pred = key_logits.float().sigmoid() > 0.5
        key_tp += (key_pred & key_tgt).sum().item()
        key_fp += (key_pred & ~key_tgt).sum().item()
        key_fn += (~key_pred & key_tgt).sum().item()
        p_pred = prev[..., N_MOUSE:] > 0.5
        p_tp += (p_pred & key_tgt).sum().item()
        p_fp += (p_pred & ~key_tgt).sum().item()
        p_fn += (~p_pred & key_tgt).sum().item()
    policy.train()

    bins = torch.cat(all_cam_bins)
    majority_acc = bins.bincount(minlength=CAMERA_BINS).max().item() / max(len(bins), 1)

    def f1(tp, fp, fn):
        return 2 * tp / max(2 * tp + fp + fn, 1)

    return {
        "cam_ce": cam_ce_sum / n_batches,
        "key_bce": key_bce_sum / n_batches,
        "cam_acc": cam_correct / max(cam_total, 1),
        "cam_acc_majority": majority_acc,
        "cam_acc_persist": cam_persist_correct / max(cam_total, 1),
        "key_f1": f1(key_tp, key_fp, key_fn),
        "key_f1_persist": f1(p_tp, p_fp, p_fn),
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    use_amp = args.amp and device.type == "cuda"

    print("=" * 78)
    print("⛏️  Minecraft VPT 离线行为克隆 (BC)")
    print(f"   data={args.data_dir} | backbone={args.backbone} | seq_len={args.seq_len} "
          f"| camera_scale={args.camera_scale} | amp={'bf16' if use_amp else 'off'}")
    print("=" * 78)

    # ─── 数据:train / holdout(独立目录优先;否则同目录按文件名切末 N 个) ───
    ds_kw = dict(seq_len=args.seq_len, img_size=args.img_size,
                 camera_scale=args.camera_scale, frame_skip=1,
                 clip_cache=args.clip_cache, seed=args.seed)
    if args.holdout_dir:
        train_ds = VPTStreamDataset(args.data_dir, split=None, **ds_kw)
        hold_ds = VPTStreamDataset(args.holdout_dir, split=None, **ds_kw)
    else:
        train_ds = VPTStreamDataset(args.data_dir, split="train",
                                    holdout_n=args.holdout_n, **ds_kw)
        hold_ds = VPTStreamDataset(args.data_dir, split="holdout",
                                   holdout_n=args.holdout_n, **ds_kw)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              num_workers=args.workers, pin_memory=True)
    hold_loader = DataLoader(hold_ds, batch_size=args.batch_size,
                             num_workers=1, pin_memory=True)
    train_iter, hold_iter = iter(train_loader), iter(hold_loader)

    # ─── 模型/优化器 ─────────────────────────────────────────────
    cfg = BCConfig(
        backbone=BackboneConfig(kind=args.backbone),
        d=args.d, heads=args.heads, layers=args.layers, dropout=args.dropout,
        max_len=max(128, args.seq_len), action_dim=ACTION_DIM,
        n_mouse=N_MOUSE, camera_bins=CAMERA_BINS,
    )
    policy = build_bc_policy(cfg).to(device)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_frozen = sum(p.numel() for p in policy.parameters()) - n_train
    print(f"✅ 模型: 可训练 {n_train/1e6:.2f}M + 冻结骨干 {n_frozen/1e6:.2f}M")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / max(args.warmup_steps, 1)))

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        missing, unexpected = policy.load_state_dict(ckpt["policy"], strict=False)
        assert not unexpected, f"checkpoint 含未知权重: {unexpected[:4]}"
        if not args.eval_only and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        print(f"♻️  已恢复 {args.resume}(step={start_step})")

    os.makedirs(args.run_dir, exist_ok=True)

    def save_ckpt(tag, step, metrics=None):
        path = os.path.join(args.run_dir, f"{tag}.pt")
        state = {k: v for k, v in policy.state_dict().items()
                 if not k.startswith("backbone.")}          # 冻结骨干不入 ckpt(HF 缓存可复原)
        torch.save({"policy": state, "optimizer": optimizer.state_dict(),
                    "step": step, "cfg": vars(args), "metrics": metrics}, path)
        print(f"💾 已保存 {path}")

    if args.eval_only:
        m = evaluate(policy, hold_iter, args.n_eval_batches, device, use_amp)
        print("📊 holdout:", {k: round(v, 4) for k, v in m.items()})
        return

    # ─── 训练循环 ─────────────────────────────────────────────────
    policy.train()
    best_loss = float("inf")
    t0 = time.time()
    try:
        for step in range(start_step, args.total_steps):
            sample = next(train_iter)
            img, prev, target = make_batch(sample, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                feats = policy.encode_frames(img)
                cam_logits, key_logits = policy(feats.float(), prev)
            cam_ce, key_bce, _ = bc_losses(cam_logits, key_logits, target)
            loss = cam_ce + args.key_loss_coeff * key_bce

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            sched.step()

            if step % args.log_interval == 0:
                sps = args.batch_size * (args.seq_len - 1) * (step - start_step + 1) \
                    / max(time.time() - t0, 1e-6)
                print(f"[{step:5d}/{args.total_steps}] loss={loss.item():.4f} "
                      f"cam_ce={cam_ce.item():.4f} key_bce={key_bce.item():.4f} "
                      f"| {sps:.0f} 帧/s")

            if (step + 1) % args.eval_interval == 0 or step + 1 == args.total_steps:
                m = evaluate(policy, hold_iter, args.n_eval_batches, device, use_amp)
                print(f"    📊 holdout@{step+1}: "
                      f"cam_acc={m['cam_acc']:.3f}(多数bin {m['cam_acc_majority']:.3f}"
                      f"/持续 {m['cam_acc_persist']:.3f}) "
                      f"key_F1={m['key_f1']:.3f}(持续 {m['key_f1_persist']:.3f}) "
                      f"cam_ce={m['cam_ce']:.3f} key_bce={m['key_bce']:.3f}")
                hold_loss = m["cam_ce"] + m["key_bce"]
                if hold_loss < best_loss:
                    best_loss = hold_loss
                    save_ckpt("best", step + 1, m)
    except KeyboardInterrupt:
        print("\n⏹️  训练中断")
    finally:
        save_ckpt("final", args.total_steps)

    print(f"✅ 完成,耗时 {(time.time()-t0)/60:.1f} 分钟,best holdout loss={best_loss:.4f}")


if __name__ == "__main__":
    main()
