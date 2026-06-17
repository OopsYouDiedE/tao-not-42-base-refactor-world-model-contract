"""MinecraftWorldModel 自监督训练入口(序列对齐的后果结构版)。

目标(数学推导见 knowledge/mental_world.md):用"初始帧编码 + 一段(图像+动作)token"去预测
**同一个未来帧的潜向量**,并要求不同上下文截止对该未来帧的预测互相一致;后果(反事实效应)加权,
不可逆事件走 𝒟 离散通道,动作只作条件输入。对齐目标取 EMA 教师 + stop-grad(JEPA,I8)。
"""
import argparse
import os
import sys
import itertools
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from domains.minecraft.vpt_dataset import VPTStreamDataset
from domains.minecraft.vpt_action import ACTION_DIM
from train.minecraft._seq import _to_float_img
from train.minecraft.losses import (
    importance_from_effect, latent_align_loss, agreement_loss, event_ce,
    noop_loss, path_invariance_loss, recon_split)
from train.minecraft.eval import evaluate
from blocks.regularization import SIGReg
from net.config import ModelConfig
from net.world_model import MinecraftWorldModel
from net.effect_tokenizer import EffectTokenizer
from utils.io import load_yaml


def _context_cutoffs(target, n):
    """在 [1, target-1] 上取至多 n 个上下文截止 k(去重升序;≥2 个才有一致性约束)。"""
    hi = max(1, target - 1)
    if n <= 1 or hi <= 1:
        return sorted({hi})
    step = (hi - 1) / (n - 1)
    return sorted({int(round(1 + i * step)) for i in range(n)})


def run_sequence(model, effect_tok, sigreg, batch_dev, cfg, beta_sigreg, amp_dev, use_amp):
    """构造时空 token 集合,对若干上下文截止预测同一未来帧 t*,返回 (total_loss, metrics)。"""
    img = batch_dev["img"]
    act_agg = batch_dev["act_agg"]
    dt = batch_dev["dt"]
    reach_id = batch_dev.get("reach_id")
    B, T = img.shape[0], img.shape[1]
    d_rev = model.d_rev

    with torch.autocast(device_type=amp_dev, enabled=use_amp):
        feats = model.extract_feats(img.reshape(B * T, *img.shape[2:]))
        z, kl = model.encode(feats)                              # [B*T,M,d], [B*T]
    M = z.shape[-2]
    z = z.view(B, T, M, model.d)
    kl = kl.view(B, T).mean()
    z_tgt = model.encode_target(feats).view(B, T, M, model.d)    # detached 教师

    # 帧单位时间(从可变 Δt 累计;喂帧不喂秒)
    tf = torch.cat([torch.zeros(B, 1, device=dt.device), dt.cumsum(dim=1)], dim=1)  # [B,T]
    target = T - 1
    act = act_agg[:, :target]                                    # 全程动作计划(条件)
    t_act = tf[:, :target]
    query_t = tf[:, target]

    z_inv0 = z[:, 0, :, d_rev:]
    z_invT = z[:, target, :, d_rev:]
    if reach_id is None:
        reach_id = torch.full((B,), -1, dtype=torch.long, device=z.device)

    cuts = _context_cutoffs(target, cfg.predictor.n_context_cutoffs)
    z_hats, acc_loss = [], torch.zeros((), device=z.device)
    metrics = {"align": 0.0, "event_ce": 0.0, "noop": 0.0, "path": 0.0,
               "commit": 0.0, "e_norm": 0.0, "recon_rev": 0.0, "recon_inv": 0.0,
               "persistence_ratio": 0.0, "event_acc": 0.0, "surprise": 0.0}

    for k in cuts:
        with torch.autocast(device_type=amp_dev, enabled=use_amp):
            out = model(z[:, :k + 1], tf[:, :k + 1], act, t_act, query_t, null=False)
            out0 = model(z[:, :k + 1], tf[:, :k + 1], act, t_act, query_t, null=True)
        e_inv = out["z_hat_inv"].float() - out0["z_hat_inv"].float()
        e_norm = e_inv.norm(dim=-1)                               # [B,M]
        w = importance_from_effect(e_norm)
        align, _ = latent_align_loss(out["z_hat"], z_tgt[:, target], w)

        event_idx, commit, _ = effect_tok(z_inv0, z_invT)
        ev_logits = out["event_logits"].mean(dim=1)              # [B,V]
        l_ev = event_ce(ev_logits, event_idx)
        l_noop = noop_loss(out["e_norm_hat"], e_norm)
        l_path = path_invariance_loss(out["z_hat_inv"], reach_id)

        acc_loss = acc_loss + (align + 0.1 * l_ev + l_noop + l_path + commit.mean())
        z_hats.append(out["z_hat"])

        rs = recon_split(out["z_hat_rev"], out["z_hat_inv"], z_tgt[:, target], d_rev)
        metrics["align"] += align.item()
        metrics["event_ce"] += l_ev.item()
        metrics["noop"] += l_noop.item()
        metrics["path"] += float(l_path.detach())
        metrics["commit"] += commit.mean().item()
        metrics["e_norm"] += e_norm.mean().item()
        metrics["surprise"] += out["surprise"].mean().item()
        metrics["event_acc"] += (ev_logits.argmax(-1) == event_idx).float().mean().item()
        for kk in ("recon_rev", "recon_inv", "persistence_ratio"):
            metrics[kk] += rs[kk]

    nk = len(cuts)
    acc_loss = acc_loss / nk
    agree = agreement_loss(z_hats)
    l_sig = sigreg(z.float().permute(2, 0, 1, 3).reshape(M, B * T, model.d))
    total = acc_loss + cfg.predictor.lambda_agree * agree + model.beta_kl * kl + beta_sigreg * l_sig

    for key in metrics:
        metrics[key] /= nk
    metrics.update(loss=total.item(), agree=agree.item(), kl=kl.item(), sigreg=l_sig.item())
    return total, metrics


def train_epoch(model, effect_tok, sigreg, data_iter, opt, scaler, device, steps,
                cfg, beta_sigreg, amp_dev, use_amp):
    model.train()
    effect_tok.train()
    agg, n = {}, 0
    for batch in itertools.islice(data_iter, steps):
        batch_dev = {
            "img": _to_float_img(batch["img"].to(device, non_blocking=True)),
            "act_agg": batch["act_agg"].to(device, non_blocking=True),
            "dt": batch["dt"].to(device, non_blocking=True),
        }
        if "reach_id" in batch:
            batch_dev["reach_id"] = batch["reach_id"][:, 0].long().to(device)

        opt.zero_grad(set_to_none=True)
        total, metrics = run_sequence(model, effect_tok, sigreg, batch_dev, cfg,
                                      beta_sigreg, amp_dev, use_amp)
        scaler.scale(total).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(effect_tok.parameters()), 1.0)
        scaler.step(opt)
        scaler.update()
        model.update_ema()
        for kk, vv in metrics.items():
            agg[kk] = agg.get(kk, 0.0) + vv
        n += 1
    return {k: v / max(n, 1) for k, v in agg.items()} | {"batches": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="runs/vpt_sample")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--steps_per_epoch", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=16)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--frame_skip", type=int, default=8)
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--clip_cache", type=int, default=4)
    ap.add_argument("--clip_refresh", type=int, default=256)
    ap.add_argument("--holdout_n", type=int, default=1)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--ckpt_dir", default="runs/mc_ckpt")
    ap.add_argument("--config", default="configs/minecraft/base.yaml")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta_sigreg", type=float, default=0.1)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = args.device
    is_cuda = str(dev).startswith("cuda")
    amp_dev = "cuda" if is_cuda else "cpu"
    use_amp = is_cuda and not args.no_amp
    print(f"=== MINECRAFT WORLD MODEL (sequence-aligned) | device={dev} | amp={use_amp} ===")

    img_size = args.img_size if args.img_size > 0 else None
    n_workers = args.workers if args.workers is not None else max(2, min(8, (os.cpu_count() or 2) - 1))

    ds = VPTStreamDataset(args.data_dir, seq_len=args.seq_len, fps=args.fps,
                          seed=args.seed, img_size=img_size, frame_skip=args.frame_skip,
                          split="train", holdout_n=args.holdout_n,
                          clip_cache=args.clip_cache, clip_refresh=args.clip_refresh)
    loader = DataLoader(ds, batch_size=args.batch, num_workers=n_workers,
                        pin_memory=is_cuda, persistent_workers=(n_workers > 0),
                        prefetch_factor=(2 if n_workers > 0 else None))
    data_iter = iter(loader)

    eval_ds = VPTStreamDataset(args.data_dir, seq_len=args.seq_len, fps=args.fps,
                               seed=args.seed + 555, img_size=img_size,
                               frame_skip=args.frame_skip, split="holdout", holdout_n=args.holdout_n)
    eval_bs = min(args.batch, 64)
    eval_batches = []

    def _get_eval_batches():
        if not eval_batches:
            it = iter(DataLoader(eval_ds, batch_size=eval_bs, num_workers=min(4, n_workers)))
            for _ in range(4):
                eval_batches.append(next(it))
        return eval_batches

    cfg = ModelConfig.from_dict(load_yaml(args.config).get("model", {}))
    cfg.max_skip = args.frame_skip

    model = MinecraftWorldModel(cfg).to(dev)
    effect_tok = EffectTokenizer(d_inv=cfg.d_inv, event_vocab_size=cfg.effect.event_vocab_size).to(dev)
    sigreg = SIGReg(knots=17, num_proj=512).to(dev)

    opt = torch.optim.Adam(list(model.parameters()) + list(effect_tok.parameters()), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_loss = None
    best_path = os.path.join(args.ckpt_dir, "best_seqalign.pt")

    for ep in range(args.epochs):
        r = train_epoch(model, effect_tok, sigreg, data_iter, opt, scaler, dev,
                        args.steps_per_epoch, cfg, args.beta_sigreg, amp_dev, use_amp)
        sched.step()
        if ep % args.log_every == 0 or ep == args.epochs - 1:
            print(f"ep {ep:4d} | loss {r['loss']:7.3f} | align {r['align']:.4f} "
                  f"agree {r['agree']:.4f} | event_acc {r['event_acc']:.2%} | e_norm {r['e_norm']:.4f} "
                  f"| recon_inv {r['recon_inv']:.4f} | sig {r['sigreg']:.2f} kl {r['kl']:.3f}")
        if args.eval_every > 0 and ((ep + 1) % args.eval_every == 0 or ep == args.epochs - 1):
            ev = evaluate(model, effect_tok, _get_eval_batches(), dev, amp_dev, use_amp, cfg)
            print(f"  [eval] align {ev['align']:.4f} | agree {ev['agree']:.4f} | "
                  f"drift {ev['rollout_drift']:.4f} | corr(w,future) {ev['corr_w_future']:.3f} "
                  f"corr(w,pixel) {ev['corr_w_pixel']:.3f}")
            if best_loss is None or ev["align"] < best_loss:
                best_loss = ev["align"]
                torch.save({
                    "model": {k: v for k, v in model.state_dict().items() if not k.startswith("backbone.")},
                    "effect_tok": effect_tok.state_dict(),
                    "epoch": ep, "align": best_loss, "config": asdict(cfg),
                }, best_path)
                print(f"  [best] align={best_loss:.4f} @ep{ep} -> {best_path}")


if __name__ == "__main__":
    main()
