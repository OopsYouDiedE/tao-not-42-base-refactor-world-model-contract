"""gaming500-720p-hdf5 上的 DINO-tokenizer 解码头预训练(design 阶段 1,冻结 DINOv3 版)。

对外接口:命令行 main。数据走 train/gaming500/dataset.Gaming500Dataset(seq_len=1,
原生图像,不经 VPT 动作契约);模型为 net/dino_tokenizer.DinoTokenizer——**冻结 DINOv3
骨干 + 可训练空间卷积解码头**,只训解码头(编码器零训练,no_grad 前向)。
设计定位见 knowledge/mental_world.md(感知借冻结 DINO)、design_gaming500_consume.md §6。

为何不从零训 net/dreamer4.Tokenizer:那条线的 decoder 有个 Linear(num_tokens·token_dim,
base_ch·min_res²),权重随 img⁴ 暴涨(img=176 时单层 34GB)必 OOM,且违背"冻结 DINO"立场。
本脚本编码器冻结、解码头直接吃 patch 空间网格上采样,无此层。

用法:
    HF_TOKEN=... PYTHONPATH=. python train/gaming500/train_tokenizer.py \
        --data-dir runs/data/g500_h5 --img-size 176 --batch-size 256 \
        --dec-depths 512,384,256,128 --cache --workers 12 --run-dir runs/g500_tok
"""
import argparse
import math
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from net.dino_tokenizer import DinoTokenizer
from train.gaming500.dataset import Gaming500Dataset


def parse_args():
    p = argparse.ArgumentParser(description="gaming500 DINO-tokenizer 解码头预训练")
    p.add_argument("--data-dir", default="runs/data/g500_h5")
    p.add_argument("--run-dir", default="runs/g500_tok")
    p.add_argument("--img-size", type=int, default=176, help="patch_size 的倍数(dinov3=16)")
    p.add_argument("--crop-mode", default="resize", choices=["resize", "center", "random"])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--total-steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--backbone", default="dinov3", choices=["dinov3", "dinov2"])
    p.add_argument("--weights", default=None, help="覆盖骨干 HF repo id(可换更大 ViT)")
    p.add_argument("--dec-depths", default="512,384,256,128",
                   help="解码头逐级通道(逗号分隔);级数须使 G·2^len==img_size")
    p.add_argument("--amp", default="bf16", choices=["off", "bf16", "fp16"])
    p.add_argument("--workers", type=int, default=os.cpu_count(),
                   help="DataLoader 进程数(默认拉满 CPU 核;每 worker cv2 单线程避免过订)")
    p.add_argument("--prefetch", type=int, default=4, help="每 worker 预取批数")
    p.add_argument("--cache", action="store_true",
                   help="全帧解码入内存(喂饱 CPU + 免训练期解码;COW 供 worker 共享)")
    p.add_argument("--holdout-frac", type=float, default=0.03)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--eval-interval", type=int, default=1000)
    p.add_argument("--n-eval-batches", type=int, default=20)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default=None)
    return p.parse_args()


def _worker_init(_):
    cv2.setNumThreads(1)                               # N 进程 × 单线程 = N 路真并行解码


def make_loader(data_dir, split, args, shuffle):
    ds = Gaming500Dataset(data_dir, seq_len=1, img_size=args.img_size,
                          crop_mode=args.crop_mode, seed=args.seed,
                          split=split, holdout_frac=args.holdout_frac,
                          cache=args.cache, cache_threads=max(2, args.workers))
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.workers, pin_memory=True,
                      persistent_workers=args.workers > 0,
                      prefetch_factor=args.prefetch if args.workers > 0 else None,
                      worker_init_fn=_worker_init, drop_last=shuffle)


def recon_step(tok, batch, device):
    """一步前向 → (loss, recon_mse)。img: batch['img'] uint8 [B,1,3,H,W]。"""
    img = batch["img"].to(device, non_blocking=True).float() / 255.0   # [B,1,3,H,W]
    img = img[:, 0]                                                     # → [B,3,H,W]
    recon, _ = tok(img)                                                # recon ∈[0,1]
    recon_mse = F.mse_loss(recon, img)
    return recon_mse, recon_mse.detach()


@torch.no_grad()
def evaluate(tok, loader, n_batches, device, amp_dtype, use_amp):
    tok.decoder.eval()
    it, mse_sum, seen = iter(loader), 0.0, 0
    for _ in range(n_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            _, mse = recon_step(tok, batch, device)
        mse_sum += mse.item()
        seen += 1
    tok.decoder.train()
    mse = mse_sum / max(seen, 1)
    return mse, -10.0 * math.log10(max(mse, 1e-10))


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    print("=" * 78, flush=True)
    print(f"🎞️  gaming500 DINO-tokenizer 解码头预训练 | 骨干={args.backbone} "
          f"img={args.img_size} crop={args.crop_mode} batch={args.batch_size} "
          f"workers={args.workers}", flush=True)
    print("=" * 78, flush=True)

    train_loader = make_loader(args.data_dir, "train", args, shuffle=True)
    hold_loader = make_loader(args.data_dir, "holdout", args, shuffle=False)

    dec_depths = tuple(int(x) for x in args.dec_depths.split(","))
    tok = DinoTokenizer(kind=args.backbone, dec_depths=dec_depths,
                        weights=args.weights).to(device)
    g = args.img_size // tok.patch_size
    assert args.img_size % tok.patch_size == 0, \
        f"--img-size 需为 patch_size({tok.patch_size}) 的倍数"
    assert g * (2 ** len(dec_depths)) == args.img_size, \
        f"dec-depths 级数 {len(dec_depths)} 不匹配:G({g})·2^L 须 == img({args.img_size})"
    n_train = sum(p.numel() for p in tok.decoder.parameters())
    n_frozen = sum(p.numel() for p in tok.backbone.parameters())
    print(f"✅ DinoTokenizer | 冻结骨干 {n_frozen/1e6:.1f}M + 解码头 {n_train/1e6:.2f}M(可训) "
          f"| patch 网格 {g}×{g}={g*g} enc_dim={tok.enc_dim}", flush=True)

    optimizer = torch.optim.AdamW(tok.decoder.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / max(args.warmup_steps, 1)))

    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        tok.decoder.load_state_dict(ck["decoder"])
        optimizer.load_state_dict(ck["optimizer"])
        start_step = ck.get("step", 0)
        print(f"♻️  已恢复 {args.resume}(step={start_step})", flush=True)

    os.makedirs(args.run_dir, exist_ok=True)

    def save_ckpt(tag, step, metrics=None):
        path = os.path.join(args.run_dir, f"{tag}.pt")
        torch.save({"decoder": tok.decoder.state_dict(),
                    "optimizer": optimizer.state_dict(), "step": step,
                    "cfg": vars(args), "metrics": metrics}, path)
        print(f"💾 已保存 {path}", flush=True)

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)
    use_amp = amp_dtype is not None and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and args.amp == "fp16")
    print(f"⚙️  混合精度: {args.amp if use_amp else 'off(fp32)'}", flush=True)

    best_psnr, t0 = -float("inf"), time.time()
    train_iter = iter(train_loader)
    tok.decoder.train()
    try:
        for step in range(start_step, args.total_steps):
            try:
                batch = next(train_iter)
            except StopIteration:                      # 单 epoch 走完(小数据)即重启迭代
                train_iter = iter(train_loader)
                batch = next(train_iter)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                loss, mse = recon_step(tok, batch, device)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(tok.decoder.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            sched.step()

            if step % args.log_interval == 0:
                fps = args.batch_size * (step - start_step + 1) \
                    / max(time.time() - t0, 1e-6)
                psnr = -10.0 * math.log10(max(mse.item(), 1e-10))
                print(f"[{step:6d}/{args.total_steps}] mse={mse.item():.5f} "
                      f"psnr={psnr:.2f}dB | {fps:.0f} 帧/s", flush=True)

            if (step + 1) % args.eval_interval == 0 or step + 1 == args.total_steps:
                mse, psnr = evaluate(tok, hold_loader, args.n_eval_batches, device,
                                     amp_dtype, use_amp)
                print(f"    📊 holdout@{step+1}: recon_psnr={psnr:.2f}dB "
                      f"mse={mse:.5f}", flush=True)
                if psnr > best_psnr:
                    best_psnr = psnr
                    save_ckpt("best", step + 1, {"psnr": psnr})
    except KeyboardInterrupt:
        print("\n⏹️  训练中断", flush=True)
    finally:
        save_ckpt("final", args.total_steps)
    print(f"✅ 完成,耗时 {(time.time()-t0)/60:.1f} 分钟,best holdout psnr={best_psnr:.2f}dB",
          flush=True)


if __name__ == "__main__":
    main()
