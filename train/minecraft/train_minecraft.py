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
    noop_loss, path_invariance_loss, recon_split, effect_guidance_loss,
    null_consequence_loss, patch_pixel_diff, pearson_corr, decorrelation_loss)
from train.minecraft.eval import evaluate
from train.minecraft.minecraft_viz import visualize_minecraft
from blocks.regularization import SIGReg
from net.config import ModelConfig
from net.world_model import MinecraftWorldModel
from net.effect_tokenizer import EffectTokenizer
from utils.io import load_yaml


def _context_cutoffs(target, n):
    """在 [max(1, target//2), target-1] 上取至多 n 个上下文截止 k。"""
    hi = max(1, target - 1)
    lo = max(1, target // 2)
    if n <= 1 or hi <= lo:
        return sorted({hi, lo} if hi != lo else {hi})[:n]
    step = (hi - lo) / (n - 1)
    return sorted({int(round(lo + i * step)) for i in range(n)})



def run_sequence(model, effect_tok, sigreg, batch_dev, cfg, beta_sigreg, beta_guide,
                 beta_decorr, amp_dev, use_amp):
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

    z_invT = z[:, target, :, d_rev:]
    if reach_id is None:
        reach_id = torch.full((B,), -1, dtype=torch.long, device=z.device)

    # 真实未来不可逆潜发散(后果目标)与逐 patch 像素差(捷径对照)——guide/诊断同口径
    fdiv = (z_tgt[:, target, :, d_rev:].float() - z_tgt[:, 0, :, d_rev:].float()).norm(dim=-1)
    pdiff = patch_pixel_diff(img, 0, target, M)                  # [B,M]

    cuts = _context_cutoffs(target, cfg.predictor.n_context_cutoffs)
    z_hats, acc_loss = [], torch.zeros((), device=z.device)
    metrics = {"align": 0.0, "align_ratio": 0.0, "event_ce": 0.0, "noop": 0.0, "path": 0.0,
               "null": 0.0, "commit": 0.0, "e_norm": 0.0, "recon_rev": 0.0, "recon_inv": 0.0,
               "persistence_ratio": 0.0, "event_acc": 0.0, "surprise": 0.0, "guide": 0.0,
               "corr_w_future": 0.0, "corr_w_pixel": 0.0}

    for k in cuts:
        with torch.autocast(device_type=amp_dev, enabled=use_amp):
            out = model(z[:, :k + 1], tf[:, :k + 1], act, t_act, query_t, null=False)
            out0 = model(z[:, :k + 1], tf[:, :k + 1], act, t_act, query_t, null=True)
        anchor_inv = z[:, k, :, d_rev:]                          # 锚=最近观测帧 inv(与 forward 锚一致)
        e_inv = out["z_hat_inv"].float() - out0["z_hat_inv"].float()
        e_norm = e_inv.norm(dim=-1)                               # [B,M]
        w = importance_from_effect(e_norm)
        align, _ = latent_align_loss(out["z_hat"], z_tgt[:, target], w)
        # 诚实技能:模型误差 / copy-last(复制锚帧)误差,<1 才算真学到前向动力学而非复制(只监控)
        me = (out["z_hat"].float() - z_tgt[:, target].float()).pow(2).mean().detach()
        ce = (z[:, k].float() - z_tgt[:, target].float()).pow(2).mean().detach()

        event_idx, commit, _ = effect_tok(anchor_inv, z_invT)
        ev_logits = out["event_logits"].mean(dim=1)              # [B,V]
        l_ev = event_ce(ev_logits, event_idx)
        l_noop = noop_loss(out["e_norm_hat"], e_norm)
        l_path = path_invariance_loss(out["z_hat_inv"], reach_id)
        l_guide = effect_guidance_loss(e_norm, fdiv, pdiff)
        l_null = null_consequence_loss(out0["z_hat_inv"], anchor_inv)  # do(null) ⇒ 回锚帧,无不可逆后果

        acc_loss = acc_loss + (align + 0.1 * l_ev + l_noop + l_path + commit.mean()
                               + beta_guide * l_guide + l_null)
        z_hats.append(out["z_hat"])

        rs = recon_split(out["z_hat_rev"], out["z_hat_inv"], z_tgt[:, target], d_rev)
        metrics["align"] += align.item()
        metrics["align_ratio"] += float(me / ce.clamp(min=1e-4))
        metrics["event_ce"] += l_ev.item()
        metrics["noop"] += l_noop.item()
        metrics["path"] += float(l_path.detach())
        metrics["null"] += l_null.item()
        metrics["commit"] += commit.mean().item()
        metrics["e_norm"] += e_norm.mean().item()
        metrics["surprise"] += out["surprise"].mean().item()
        metrics["event_acc"] += (ev_logits.argmax(-1) == event_idx).float().mean().item()
        metrics["guide"] += l_guide.item()
        # 训练侧也记 corr 诊断(w/fdiv/pdiff 均 detach,只监控不反传),直接看 train↔eval gap
        metrics["corr_w_future"] += float(pearson_corr(w, fdiv))
        metrics["corr_w_pixel"] += float(pearson_corr(w, pdiff))
        for kk in ("recon_rev", "recon_inv", "persistence_ratio"):
            metrics[kk] += rs[kk]

    nk = len(cuts)
    acc_loss = acc_loss / nk
    agree = agreement_loss(z_hats)
    l_sig = sigreg(z.float().permute(2, 0, 1, 3).reshape(M, B * T, model.d))
    l_decorr = decorrelation_loss(z[..., :d_rev], z[..., d_rev:])  # z_rev⊥z_inv 断外观泄漏
    total = (acc_loss + cfg.predictor.lambda_agree * agree + model.beta_kl * kl
             + beta_sigreg * l_sig + beta_decorr * l_decorr)

    for key in metrics:
        metrics[key] /= nk
    metrics.update(loss=total.item(), agree=agree.item(), kl=kl.item(),
                   sigreg=l_sig.item(), decorr=l_decorr.item())
    return total, metrics



def train_epoch(model, effect_tok, sigreg, data_iter, opt, scaler, device, steps,
                cfg, beta_sigreg, beta_guide, beta_decorr, amp_dev, use_amp):
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
                                      beta_sigreg, beta_guide, beta_decorr, amp_dev, use_amp)
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
    ap.add_argument("--holdout_dir", default=None, help="独立测试集目录(滚动模式下必传)")
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
    ap.add_argument("--viz_every", type=int, default=0, help="每多少 epoch 渲染一张诊断面板(0=关闭)")
    ap.add_argument("--viz_dir", default="runs/mc_viz", help="诊断面板 PNG 输出目录")
    ap.add_argument("--ckpt_dir", default="runs/mc_ckpt")
    ap.add_argument("--config", default="configs/minecraft/base.yaml")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta_sigreg", type=float, default=0.1)
    ap.add_argument("--beta_guide", type=float, default=0.1,
                    help="后果权重引导损失权重(corr(w,future)↑/corr(w,pixel)→0;0=关闭)")
    ap.add_argument("--beta_decorr", type=float, default=0.1,
                    help="z_rev⊥z_inv 跨通道去相关权重(断不可逆通道的外观泄漏;0=关闭)")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", action="store_true", help="启用 W&B 实验同步")
    ap.add_argument("--wandb_project", default="minecraft-world-model", help="W&B 项目名")
    ap.add_argument("--wandb_run", default=None, help="W&B 运行实例名")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = args.device
    is_cuda = str(dev).startswith("cuda")
    amp_dev = "cuda" if is_cuda else "cpu"
    use_amp = is_cuda and not args.no_amp
    print(f"=== MINECRAFT WORLD MODEL (sequence-aligned) | device={dev} | amp={use_amp} ===")

    img_size = args.img_size if args.img_size > 0 else None
    n_workers = args.workers if args.workers is not None else max(2, min(8, (os.cpu_count() or 2) - 1))

    split_train = None if args.holdout_dir else "train"
    ds = VPTStreamDataset(args.data_dir, seq_len=args.seq_len, fps=args.fps,
                          seed=args.seed, img_size=img_size, frame_skip=args.frame_skip,
                          split=split_train, holdout_n=args.holdout_n,
                          clip_cache=args.clip_cache, clip_refresh=args.clip_refresh)
    loader = DataLoader(ds, batch_size=args.batch, num_workers=n_workers,
                        pin_memory=is_cuda, persistent_workers=(n_workers > 0),
                        prefetch_factor=(2 if n_workers > 0 else None))
    data_iter = iter(loader)

    eval_dir = args.holdout_dir if args.holdout_dir else args.data_dir
    split_eval = None if args.holdout_dir else "holdout"
    eval_ds = VPTStreamDataset(eval_dir, seq_len=args.seq_len, fps=args.fps,
                               seed=args.seed + 555, img_size=img_size,
                               frame_skip=args.frame_skip, split=split_eval, holdout_n=args.holdout_n)
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

    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            config={
                "lr": args.lr,
                "epochs": args.epochs,
                "batch": args.batch,
                "seq_len": args.seq_len,
                "frame_skip": args.frame_skip,
                "model_config": asdict(cfg),
            }
        )

    opt = torch.optim.Adam(list(model.parameters()) + list(effect_tok.parameters()), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_loss = None
    best_path = os.path.join(args.ckpt_dir, "best_seqalign.pt")

    for ep in range(args.epochs):
        r = train_epoch(model, effect_tok, sigreg, data_iter, opt, scaler, dev,
                        args.steps_per_epoch, cfg, args.beta_sigreg, args.beta_guide,
                        args.beta_decorr, amp_dev, use_amp)
        sched.step()
        if ep % args.log_every == 0 or ep == args.epochs - 1:
            print(f"ep {ep:4d} | loss {r['loss']:7.3f} | align_ratio {r['align_ratio']:.3f} "
                  f"agree {r['agree']:.4f} | e_norm {r['e_norm']:.4f} "
                  f"| guide {r['guide']:.4f} null {r['null']:.4f} "
                  f"corr(w,fut) {r['corr_w_future']:+.3f} | sig {r['sigreg']:.2f}")

        # W&B 训练面板:精简到 9 条核心指标(诚实技能/一致/防坍/反捷径/去相关)
        if args.wandb:
            log_dict = {
                "epoch": ep,
                "train/loss": r["loss"],
                "train/align_ratio": r["align_ratio"],
                "train/agree": r["agree"],
                "train/e_norm": r["e_norm"],
                "train/sigreg": r["sigreg"],
                "train/guide": r["guide"],
                "train/null": r["null"],
                "train/corr_w_future": r["corr_w_future"],
                "train/decorr": r["decorr"],
            }

        if args.eval_every > 0 and ((ep + 1) % args.eval_every == 0 or ep == args.epochs - 1):
            ev = evaluate(model, effect_tok, _get_eval_batches(), dev, amp_dev, use_amp, cfg)
            print(f"  [eval] align_ratio {ev['align_ratio']:.3f} | agree {ev['agree']:.4f} | "
                  f"drift {ev['rollout_drift']:.4f}")

            # W&B 评估面板:精简到 3 条独立健康轴(诚实技能比/多上下文一致/闭环漂移)
            if args.wandb:
                log_dict.update({
                    "eval/align_ratio": ev["align_ratio"],
                    "eval/agree": ev["agree"],
                    "eval/rollout_drift": ev["rollout_drift"],
                })

            if best_loss is None or ev["align_ratio"] < best_loss:
                best_loss = ev["align_ratio"]
                torch.save({
                    "model": {k: v for k, v in model.state_dict().items() if not k.startswith("backbone.")},
                    "effect_tok": effect_tok.state_dict(),
                    "epoch": ep, "align_ratio": best_loss, "config": asdict(cfg),
                }, best_path)
                print(f"  [best] align_ratio={best_loss:.3f} @ep{ep} -> {best_path}")
                
                # 记录最好模型权重到 W&B Artifacts
                if args.wandb:
                    artifact = wandb.Artifact(name=f"best-model-{args.wandb_run or 'run'}", type="model")
                    artifact.add_file(best_path)
                    wandb.log_artifact(artifact)
                    print(f"  [wandb] Best checkpoint uploaded as artifact: {artifact.name}")

        if args.viz_every > 0 and ((ep + 1) % args.viz_every == 0 or ep == args.epochs - 1):
            os.makedirs(args.viz_dir, exist_ok=True)
            png = os.path.join(args.viz_dir, f"ep{ep:04d}.png")
            visualize_minecraft(model, effect_tok, _get_eval_batches()[0], cfg,
                                dev, amp_dev, use_amp, png, epoch=ep)
            print(f"  [viz] -> {png}")
            if args.wandb:
                log_dict["viz/panel"] = wandb.Image(png)

        if args.wandb:
            wandb.log(log_dict)


if __name__ == "__main__":
    main()
