"""MinecraftWorldModel 自监督训练的命令行入口 (两步式离散词表与重构版)。
"""
import argparse
import os
import sys
import itertools
import time
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 引入路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from domains.minecraft.vpt_dataset import VPTStreamDataset
from domains.minecraft.vpt_action import ACTION_DIM
from train.minecraft._seq import roll_hist, _to_float_img
from train.minecraft.losses import vocab_pred_loss, z_recon_loss
from train.minecraft.eval import evaluate
from blocks.regularization import SIGReg
from net.config import ModelConfig
from net.world_model import MinecraftWorldModel
from net.action_model import ActionTokenizer
from utils.io import load_yaml


def _run_sequence(model, action_tok, sigreg, batch_dev, device, k_bptt,
                  beta_sigreg, beta_recon, amp_dev, use_amp, scaler, acc):
    """两步式截断 BPTT 训练单步前向与反向。"""
    img, t_vec = batch_dev["img"], batch_dev["t_vec"]
    act_seq, dt = batch_dev["act_seq"], batch_dev["dt"]
    task_emb = batch_dev.get("task_emb")
    B, T = img.shape[0], img.shape[1]
    
    h = torch.zeros(B, 1, model.d, device=device)
    # 历史动作使用聚合动作历史：对历史区间内的动作取平均
    a_hist = torch.zeros(B, model.J, ACTION_DIM, device=device)
    t_hist = torch.zeros(B, model.J, device=device)
    hv = torch.zeros(B, model.J, device=device)
    n_win = -(-(T - 1) // k_bptt)

    # 提取特征
    with torch.autocast(device_type=amp_dev, enabled=use_amp):
        feats = model.extract_feats(img.reshape(B * T, *img.shape[2:]))
        # z_tg 形状是 [B, T, M, d]
        z_tg = model.encode_obs(feats=feats).view(B, T, -1, model.d)
    feats = feats.view(B, T, *feats.shape[-2:])
    z_tg = z_tg.float()

    for w0 in range(0, T - 1, k_bptt):
        w1 = min(w0 + k_bptt, T - 1)
        k = w1 - w0
        z_obs = z_tg[:, w0:w1]

        accum = torch.zeros((), device=device)
        for i in range(k):
            t = w0 + i
            
            with torch.autocast(device_type=amp_dev, enabled=use_amp):
                # 1. 提取 GT 离散 Token ID（使用 ActionTokenizer 量化压缩连续动作）
                valid = (torch.arange(model.S, device=device).unsqueeze(0) < dt[:, t].unsqueeze(1)).float()
                _, target_token_id, tok_loss = action_tok(act_seq[:, t], valid_mask=valid)
                
                # 2. 世界模型前向推演
                out = model(
                    z_obs[:, i], h, a_hist, act_seq[:, t], dt[:, t], t_vec[:, t],
                    t_hist=t_hist, hist_valid=hv, task_emb=task_emb,
                    target_token_id=target_token_id
                )

            # 3. 滚动历史
            agg_action = act_seq[:, t].mean(dim=1)  # 区间动作均值作为代表动作滚入历史
            a_hist, t_hist, hv = roll_hist(a_hist, t_hist, hv, agg_action, dt[:, t])

            # 4. 损失计算 (FP32)
            logits = out["logits"].float()
            z_recon = out["z_recon"].float()
            
            l_vocab = vocab_pred_loss(logits, target_token_id)
            l_recon, r_recon = z_recon_loss(z_recon, z_tg[:, t + 1])
            l_tok = tok_loss.mean()
            
            step_loss = l_vocab + beta_recon * l_recon + l_tok
            accum = accum + step_loss / (T - 1)
            h = out["h_next"]

            # 指标记录
            acc["vocab_loss"] += l_vocab.detach()
            acc["recon_loss"] += l_recon.detach()
            acc["recon_ratio"] += r_recon
            acc["vocab_acc"] += (logits.argmax(dim=-1) == target_token_id).float().mean().detach()
            acc["tok_loss"] += l_tok.detach()
            acc["inner"] += 1

        # SIGReg 防坍缩正则化
        l_sig = sigreg(z_obs.float().permute(2, 1, 0, 3).reshape(z_obs.shape[2], k * B, model.d))
        win_loss = accum + (beta_sigreg / n_win) * l_sig
        
        scaler.scale(win_loss).backward()
        acc["loss"] += win_loss.detach()
        acc["sigreg"] += l_sig.detach()
        acc["win"] += 1
        h = h.detach()


def train_epoch(model, action_tok, sigreg, data_iter, opt, scaler, device, steps, k_bptt,
                beta_sigreg, beta_recon, text_enc, amp_dev, use_amp):
    """进行一个 epoch 的训练。"""
    model.train()
    action_tok.train()
    
    acc = {k: torch.zeros((), device=device) for k in
           ["loss", "vocab_loss", "recon_loss", "recon_ratio", "vocab_acc", "tok_loss", "sigreg"]}
    acc["inner"] = acc["win"] = 0
    n_batches = 0

    for batch in itertools.islice(data_iter, steps):
        batch_dev = {
            "img": _to_float_img(batch["img"].to(device, non_blocking=True)),
            "act_seq": batch["act_seq"].to(device, non_blocking=True),
            "dt": batch["dt"].to(device, non_blocking=True),
            "t_vec": batch["t_vec"].to(device, non_blocking=True),
            "task_emb": (text_enc.encode(batch["task_text"]).to(device)
                         if text_enc is not None else None),
        }

        opt.zero_grad(set_to_none=True)
        _run_sequence(
            model, action_tok, sigreg, batch_dev, device, k_bptt,
            beta_sigreg, beta_recon, amp_dev, use_amp, scaler, acc
        )
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(action_tok.parameters()), 1.0)
        scaler.step(opt)
        scaler.update()
        n_batches += 1

    ic = max(acc["inner"], 1)
    wc = max(acc["win"], 1)
    return {
        "loss": (acc["loss"] / wc).item(),
        "vocab_loss": (acc["vocab_loss"] / ic).item(),
        "recon_loss": (acc["recon_loss"] / ic).item(),
        "recon_ratio": (acc["recon_ratio"] / ic).item(),
        "vocab_acc": (acc["vocab_acc"] / ic).item(),
        "tok_loss": (acc["tok_loss"] / ic).item(),
        "sigreg": (acc["sigreg"] / wc).item(),
        "batches": n_batches,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="runs/vpt_sample", help="训练数据目录")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--steps_per_epoch", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=60)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--frame_skip", type=int, default=8)
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--cache_size", type=int, default=32)
    ap.add_argument("--clip_cache", type=int, default=4)
    ap.add_argument("--clip_refresh", type=int, default=256)
    ap.add_argument("--holdout_n", type=int, default=1)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--ckpt_dir", default="runs/mc_ckpt")
    ap.add_argument("--config", default="configs/minecraft/base.yaml")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta_recon", type=float, default=1.0, help="重构损失权重")
    ap.add_argument("--beta_sigreg", type=float, default=0.1, help="SIGReg 正则权重")
    ap.add_argument("--k_bptt", type=int, default=4)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--text_encoder", choices=["minilm", "mock", "none"], default="minilm")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = args.device
    is_cuda = str(dev).startswith("cuda")
    amp_dev = "cuda" if is_cuda else "cpu"
    use_amp = is_cuda and not args.no_amp
    
    print(f"=== MINECRAFT WORLD MODEL (Two-Step Clean Version) | device={dev} | amp={use_amp} ===")

    # 数据加载
    text_enc = None if args.text_encoder == "none" else TaskTextEncoder(args.text_encoder, device="cpu")
    img_size = args.img_size if args.img_size > 0 else None
    n_workers = args.workers if args.workers is not None else max(2, min(8, (os.cpu_count() or 2) - 1))
    
    ds = VPTStreamDataset(args.data_dir, seq_len=args.seq_len, fps=args.fps,
                          cache_size=args.cache_size, seed=args.seed, img_size=img_size,
                          frame_skip=args.frame_skip, split="train", holdout_n=args.holdout_n,
                          clip_cache=args.clip_cache, clip_refresh=args.clip_refresh)
    
    loader = DataLoader(ds, batch_size=args.batch, num_workers=n_workers,
                        pin_memory=is_cuda, persistent_workers=(n_workers > 0),
                        prefetch_factor=(2 if n_workers > 0 else None))
    data_iter = iter(loader)

    # 评估数据集
    eval_ds = VPTStreamDataset(args.data_dir, seq_len=args.seq_len, fps=args.fps,
                               cache_size=4, seed=args.seed + 555, img_size=img_size,
                               frame_skip=args.frame_skip, split="holdout", holdout_n=args.holdout_n)
    eval_bs = min(args.batch, 64)
    eval_batches = []
    
    def _get_eval_batches():
        if not eval_batches:
            it = iter(DataLoader(eval_ds, batch_size=eval_bs, num_workers=min(4, n_workers)))
            for _ in range(4):
                b = next(it)
                if text_enc is not None:
                    b["task_emb"] = text_enc.encode(b["task_text"])
                eval_batches.append(b)
        return eval_batches

    # 加载模型配置并实例化模型与动作分词器
    cfg = ModelConfig.from_dict(load_yaml(args.config).get("model", {}))
    cfg.max_skip = args.frame_skip
    
    model = MinecraftWorldModel(cfg).to(dev)
    action_tok = ActionTokenizer(act_dim=ACTION_DIM, hidden_dim=128, latent_dim=128, n_embed=512).to(dev)

    sigreg = SIGReg(knots=17, num_proj=512).to(dev)
    
    # 联合优化
    opt = torch.optim.Adam(list(model.parameters()) + list(action_tok.parameters()), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_loss = None
    best_path = os.path.join(args.ckpt_dir, "best_two_step.pt")

    for ep in range(args.epochs):
        r = train_epoch(
            model, action_tok, sigreg, data_iter, opt, scaler, dev,
            args.steps_per_epoch, args.k_bptt, args.beta_sigreg, args.beta_recon,
            text_enc, amp_dev, use_amp
        )
        sched.step()

        if ep % args.log_every == 0 or ep == args.epochs - 1:
            print(f"ep {ep:4d} | loss {r['loss']:7.3f} | vocab_loss {r['vocab_loss']:.3f} "
                  f"vocab_acc {r['vocab_acc']:.2%} | recon_loss {r['recon_loss']:.4f} "
                  f"recon_ratio {r['recon_ratio']:.3f} | sig {r['sigreg']:.2f}")

        # 周期性 holdout 评估
        if args.eval_every > 0 and ((ep + 1) % args.eval_every == 0 or ep == args.epochs - 1):
            ev = evaluate(model, action_tok, _get_eval_batches(), dev, amp_dev, use_amp)
            print(f"  [eval] loss {ev['loss']:.3f} | vocab_acc {ev['vocab_acc']:.2%} | recon_ratio {ev['recon_ratio']:.3f}")
            
            # 保存 best checkpoint
            if best_loss is None or ev["loss"] < best_loss:
                best_loss = ev["loss"]
                torch.save({
                    "model": {k: v for k, v in model.state_dict().items() if not k.startswith("backbone.")},
                    "action_tok": action_tok.state_dict(),
                    "epoch": ep,
                    "loss": best_loss,
                    "config": asdict(cfg)
                }, best_path)
                print(f"  [best] loss={best_loss:.4f} @ep{ep} -> {best_path}")


if __name__ == "__main__":
    main()
