"""RSSM + 后继特征世界模型的最小训练/评估入口(设计见 knowledge/rssm_sf_design.md)。

切片目标:证明两条可证伪机制成立,而非生产规模。
  1) 难 horizon align_ratio < 1:后验观测到 k=T//2,先验开环滚到 T-1,grounding 头解码 ê_{T-1},
     与 copy-last(e_k)比 ——旧 anchor+有界增量模型在此 horizon 必败。
  2) ψ dose-response:holdout 上 corr(ψ_t, D_t),D_t=Σ_k γ^k φ_{t+k}(经验折扣未来 has_item)。

感知前端 = 冻结骨干 + patch 池化 → 固定嵌入 e(grounding 目标,不参与梯度 ⇒ 无坍缩平凡解)。
本文件在 train/(允许 import domain、读文件);骨干 mock 经依赖注入只在 tests/(AGENTS §2)。
"""
import argparse
import copy
import os
import sys
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from net.backbone import build_backbone
from net.config import ModelConfig, RSSMConfig
from net.rssm import RSSM, EPS
from train.minecraft._seq import _to_float_img
from train.minecraft.losses import pearson_corr
from domains.minecraft.vpt_action import ACTION_DIM
from domains.minecraft.vpt_dataset import VPTStreamDataset
from utils.io import load_yaml


class FrozenBackbonePerception(nn.Module):
    """冻结视觉骨干 + patch 池化 → 帧级固定嵌入 e。

    forward(img[B,3,H,W] float∈[0,1]) → e[B,E]。骨干无梯度、eval 模式;DINOv2/v3 走
    归一化+整除补齐+丢 CLS/register;注入 mock(kind=injected)直接取 patch token 均值。
    """

    def __init__(self, backbone_cfg, injected=None):
        super().__init__()
        self.backbone, self._patch, self.embed_dim, self._n_reg, self.kind = \
            build_backbone(backbone_cfg, injected)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, img):
        if self.kind in ("dinov2", "dinov3"):
            H, W = img.shape[-2:]
            ps = self._patch
            H2, W2 = max(ps, (H // ps) * ps), max(ps, (W // ps) * ps)
            if (H2, W2) != (H, W):
                img = F.interpolate(img, size=(H2, W2), mode="bilinear", align_corners=False)
            img = (img - self._mean) / self._std
            tokens = self.backbone(pixel_values=img).last_hidden_state[:, 1 + self._n_reg:, :]
        else:
            tokens = self.backbone(img)                      # mock: [B,M,E]
        return tokens.float().mean(dim=1)                    # [B,E] patch 均值池化

    def encode_seq(self, img_seq):
        """img_seq[B,T,3,H,W] → e[B,T,E](逐帧编码,detach)。"""
        B, T = img_seq.shape[0], img_seq.shape[1]
        e = self(img_seq.reshape(B * T, *img_seq.shape[2:]))
        return e.view(B, T, -1).detach()


# ---- 后继特征 TD(λ) 目标 ----
def lambda_return_sf(phi, psi_tgt, gamma, lam):
    """后继特征的 λ-return 目标。

    phi[B,T,F]、psi_tgt[B,T,F](目标网在同 feats 上的 ψ)→ targets[B,T-1,F]。
    G_t = φ_t + γ[(1-λ)ψ_{t+1} + λ G_{t+1}],末帧用 ψ_tgt 自举(SR Bellman:ψ=φ+γψ')。
    """
    T = phi.shape[1]
    g_next = psi_tgt[:, -1]
    outs = []
    for t in reversed(range(T - 1)):
        g = phi[:, t] + gamma * ((1.0 - lam) * psi_tgt[:, t + 1] + lam * g_next)
        outs.append(g)
        g_next = g
    outs.reverse()
    return torch.stack(outs, dim=1)


def empirical_discounted_future(phi, gamma):
    """经验截断折扣未来 D_t=Σ_{k} γ^k φ_{t+k}(窗内,末端无自举)。phi[B,T,F]→D[B,T,F]。"""
    T = phi.shape[1]
    outs = [None] * T
    outs[T - 1] = phi[:, T - 1]
    for t in reversed(range(T - 1)):
        outs[t] = phi[:, t] + gamma * outs[t + 1]
    return torch.stack(outs, dim=1)


# ---- 训练步(可测)----
def rssm_loss(rssm, sf_target, e, actions, phi, phi_mask, gamma, lam, beta_ground):
    """一次前向 + 三损失(KL / grounding / 后继特征 TD)。

    e[B,T,E]、actions[B,T-1,A]、phi[B,T,F](-1 已置 0)、phi_mask[B,T,F]∈{0,1}。
    返回 (total 标量, metrics dict, feats[B,T,feat_dim])。
    """
    feats, post, prior, _ = rssm.observe(e, actions)
    kl, kl_value = rssm.kl_loss(post, prior)
    e_hat = rssm.grounding_head(feats)
    ground = F.mse_loss(e_hat, e)                            # decoder-free 重建(目标冻结)

    if phi is not None:                                      # 有 GT 才训后继特征(无 GT 仍训 RSSM+grounding)
        psi = rssm.sf_head(feats)                            # [B,T,F]
        with torch.no_grad():
            psi_tgt = sf_target(feats)                       # 目标网自举值
        targets = lambda_return_sf(phi, psi_tgt, gamma, lam).detach()   # [B,T-1,F]
        m = phi_mask[:, :-1]
        sf_err = F.smooth_l1_loss(psi[:, :-1], targets, reduction="none") * m
        sf = sf_err.sum() / m.sum().clamp(min=1.0)
    else:
        sf = torch.zeros((), device=e.device)

    total = kl + beta_ground * ground + sf
    metrics = {"loss": total.item(), "kl": kl_value.item(), "ground": ground.item(),
               "sf": sf.item()}
    return total, metrics, feats


@torch.no_grad()
def hard_horizon_align_ratio(rssm, e, actions):
    """难 horizon 诚实技能比(验收线 1)。

    观测到 k=T//2,先验开环滚到 T-1,grounding 解码 ê_{T-1};ratio=‖ê-e_{T-1}‖²/‖e_k-e_{T-1}‖²。
    返回 (ratio float, k)。<1 = 在难 horizon 上赢 copy-last。
    """
    B, T = e.shape[0], e.shape[1]
    k = T // 2
    # 评估走确定性 rollout(均值/众数):消除采样方差与 MSE 上偏,给前向技能干净点估计
    _, _, _, states = rssm.observe(e[:, :k + 1], actions[:, :k], sample=False)
    state_k = {"h": states["h"][:, k], "z": states["z"][:, k]}
    img_feats = rssm.imagine(state_k, actions[:, k:T - 1], sample=False)   # 帧 k+1..T-1
    e_hat = rssm.grounding_head(img_feats[:, -1])            # ê_{T-1}
    num = (e_hat - e[:, T - 1]).pow(2).mean(dim=-1)
    den = (e[:, k] - e[:, T - 1]).pow(2).mean(dim=-1)        # copy-last 基线
    return float((num / den.clamp(min=EPS)).mean()), k


@torch.no_grad()
def dose_response_corr(rssm, e, actions, phi, phi_mask, gamma):
    """ψ dose-response(验收线 2):corr(ψ_t, 经验折扣未来 D_t),masked。无 GT 返回 nan。"""
    if phi is None:
        return float("nan")
    feats, _, _, _ = rssm.observe(e, actions, sample=False)   # 确定性后验,稳定读 ψ
    psi = rssm.sf_head(feats)
    D = empirical_discounted_future(phi, gamma)
    m = phi_mask.bool()
    if m.sum() < 2:
        return float("nan")
    return float(pearson_corr(psi[m], D[m]))


@torch.no_grad()
def update_target(target_head, live_head, decay):
    for tp, lp in zip(target_head.parameters(), live_head.parameters()):
        tp.mul_(decay).add_(lp.detach(), alpha=1.0 - decay)


# ---- 数据装配 ----
def _prep_batch(batch, perception, device):
    """batch → (e[B,T,E], actions[B,T-1,A], phi, phi_mask)。

    无 has_item GT(真 BASALT)时 phi=phi_mask=None:只训 RSSM+grounding,验收线 1(align_hard)
    仍可测;验收线 2(ψ dose-response)需 GT(download_sample_data --counterfactual)才激活。
    """
    img = _to_float_img(batch["img"].to(device))
    actions = batch["act_agg"].to(device)
    e = perception.encode_seq(img)
    if "has_item" not in batch:
        return e, actions, None, None
    raw = batch["has_item"].to(device).float()               # [B,T];-1=缺标
    phi_mask = (raw >= 0).float().unsqueeze(-1)               # [B,T,1]
    phi = raw.clamp(min=0).unsqueeze(-1)                      # [B,T,1](-1→0)
    return e, actions, phi, phi_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="runs/vpt_sample")
    ap.add_argument("--holdout_dir", default=None)
    ap.add_argument("--config", default="configs/minecraft/dinov2.yaml")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--steps_per_epoch", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=32)
    ap.add_argument("--frame_skip", type=int, default=8)
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.97)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--beta_ground", type=float, default=1.0)
    ap.add_argument("--free_nats", type=float, default=1.0)
    ap.add_argument("--target_decay", type=float, default=0.98)
    ap.add_argument("--clip_cache", type=int, default=4)
    ap.add_argument("--clip_refresh", type=int, default=256)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", default="minecraft-world-model")
    ap.add_argument("--wandb_run", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = args.device
    is_cuda = str(dev).startswith("cuda")
    n_workers = max(2, min(8, (os.cpu_count() or 2) - 1))
    print(f"=== RSSM + Successor Features | device={dev} ===")

    model_cfg = ModelConfig.from_dict(load_yaml(args.config).get("model", {}))
    perception = FrozenBackbonePerception(model_cfg.backbone).to(dev)
    rssm_cfg = RSSMConfig(embed_dim=perception.embed_dim, act_dim=ACTION_DIM,
                          free_nats=args.free_nats, sf_dim=1)
    rssm = RSSM(rssm_cfg).to(dev)
    sf_target = copy.deepcopy(rssm.sf_head)
    for p in sf_target.parameters():
        p.requires_grad_(False)

    split_train = None if args.holdout_dir else "train"
    ds = VPTStreamDataset(args.data_dir, seq_len=args.seq_len, fps=args.fps, seed=args.seed,
                          img_size=args.img_size, frame_skip=args.frame_skip, split=split_train,
                          clip_cache=args.clip_cache, clip_refresh=args.clip_refresh)
    loader = DataLoader(ds, batch_size=args.batch, num_workers=n_workers,
                        pin_memory=is_cuda, persistent_workers=(n_workers > 0),
                        prefetch_factor=(2 if n_workers > 0 else None))
    data_iter = iter(loader)

    eval_dir = args.holdout_dir or args.data_dir
    split_eval = None if args.holdout_dir else "holdout"
    eval_ds = VPTStreamDataset(eval_dir, seq_len=args.seq_len, fps=args.fps, seed=args.seed + 555,
                               img_size=args.img_size, frame_skip=args.frame_skip, split=split_eval)
    eval_batches = []

    def _eval_batches():
        if not eval_batches:
            it = iter(DataLoader(eval_ds, batch_size=min(args.batch, 64), num_workers=min(4, n_workers)))
            for _ in range(4):
                eval_batches.append(next(it))
        return eval_batches

    opt = torch.optim.Adam(rssm.parameters(), lr=args.lr)
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run,
                   config={**vars(args), "rssm": vars(rssm_cfg)})

    for ep in range(args.epochs):
        rssm.train()
        agg, cnt = {}, {}
        for batch in itertools.islice(data_iter, args.steps_per_epoch):
            e, actions, phi, phi_mask = _prep_batch(batch, perception, dev)
            opt.zero_grad(set_to_none=True)
            total, metrics, _ = rssm_loss(rssm, sf_target, e, actions, phi, phi_mask,
                                          args.gamma, args.lam, args.beta_ground)
            total.backward()
            torch.nn.utils.clip_grad_norm_(rssm.parameters(), 1.0)
            opt.step()
            update_target(sf_target, rssm.sf_head, args.target_decay)
            metrics["align_hard"], _ = hard_horizon_align_ratio(rssm, e, actions)
            metrics["dose_corr"] = dose_response_corr(rssm, e, actions, phi, phi_mask, args.gamma)
            for k, v in metrics.items():
                if v != v:                                   # 跳过 nan(无 GT 时的 dose_corr)
                    continue
                agg[k] = agg.get(k, 0.0) + v
                cnt[k] = cnt.get(k, 0) + 1
        r = {k: agg[k] / max(cnt[k], 1) for k in agg}

        if ep % args.log_every == 0:
            print(f"ep {ep:4d} | loss {r['loss']:.3f} | kl {r['kl']:.3f} ground {r['ground']:.3f} "
                  f"sf {r['sf']:.4f} | align_hard {r['align_hard']:.3f} (<1=win) "
                  f"dose_corr {r.get('dose_corr', float('nan')):+.3f}")
        log = {"epoch": ep, **{f"train/{k}": v for k, v in r.items()}}

        if args.eval_every > 0 and ((ep + 1) % args.eval_every == 0 or ep == args.epochs - 1):
            rssm.eval()
            ar, nb, dc, dn = 0.0, 0, 0.0, 0
            for b in _eval_batches():
                e, actions, phi, phi_mask = _prep_batch(b, perception, dev)
                a, _ = hard_horizon_align_ratio(rssm, e, actions)
                d = dose_response_corr(rssm, e, actions, phi, phi_mask, args.gamma)
                ar += a; nb += 1
                if d == d:                                   # 有 GT 才计 dose_corr
                    dc += d; dn += 1
            ar = ar / max(nb, 1)
            dc = dc / dn if dn else float("nan")
            print(f"  [eval] align_hard {ar:.3f} (<1=win) | dose_corr {dc:+.3f}")
            log["eval/align_hard"] = ar
            if dn:
                log["eval/dose_corr"] = dc

        if args.wandb:
            wandb.log(log)


if __name__ == "__main__":
    main()
