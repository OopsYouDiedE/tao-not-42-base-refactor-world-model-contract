"""离线诊断评估(序列对齐版):潜对齐 / 多上下文一致 / 闭环漂移 / 反捷径去相关。

反捷径主指标(数学 (8)):后果权重 w 应与**下游潜发散**正相关、与**像素差**去相关
——证明模型按"后果"而非"像素"分配重要性,没在编码上偷懒。
"""
import math

import torch

from train.minecraft._seq import _to_float_img
from train.minecraft.losses import (
    importance_from_effect, latent_align_loss, agreement_loss)


def _pearson(a, b, eps=1e-4):
    """两个 1D 张量的 Pearson 相关(fp32,I4;分母 clamp,I1)。"""
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp(min=eps)
    return float((a @ b) / denom)


def linear_probe_acc(X, y, steps=300, lr=0.5, eps=1e-4):
    """防假成功探针:从 z_inv 线性解码二值 GT(has_item/airborne)的可分性。

    X: [N, d_inv](stop-grad 输入,只训探针);y: [N] ∈{0,1}。返回训练集准确率 ∈[0,1]。
    探针失败(≈0.5)= z_inv 把小而关键的位丢了 → 触发 Phase A 升级(部分解冻 backbone)。
    """
    X = X.float().detach()
    y = (y.float().detach() > 0.5).float()
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        logit = X @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = ((X @ w + b) > 0).float()
        return float((pred == y).float().mean())


def _patch_pixel_diff(img, anchor, target, M):
    """anchor/target 两帧的逐 patch 像素差幅度 → [B, M]。"""
    B = img.shape[0]
    hw = int(round(math.sqrt(M)))
    H, W = (hw, hw) if hw * hw == M else (1, M)
    diff = (img[:, target].float() - img[:, anchor].float()).abs().mean(dim=1, keepdim=True)  # [B,1,h,w]
    pooled = torch.nn.functional.adaptive_avg_pool2d(diff, (H, W))   # [B,1,H,W]
    return pooled.reshape(B, M)


@torch.no_grad()
def evaluate(model, effect_tok, loader, device, amp_dev, use_amp, cfg):
    """holdout 上评估对齐/一致/闭环漂移/反捷径去相关。loader 为 batch 可迭代对象。"""
    model.eval()
    effect_tok.eval()
    d_rev = model.d_rev
    tot = {"align": 0.0, "agree": 0.0, "rollout_drift": 0.0,
           "corr_w_future": 0.0, "corr_w_pixel": 0.0}
    n = 0

    for batch in loader:
        img = _to_float_img(batch["img"].to(device))
        act_agg = batch["act_agg"].to(device)
        dt = batch["dt"].to(device)
        B, T = img.shape[0], img.shape[1]
        if T < 3:
            continue

        with torch.autocast(device_type=amp_dev, enabled=use_amp):
            feats = model.extract_feats(img.reshape(B * T, *img.shape[2:]))
            z, _ = model.encode(feats)
        M = z.shape[-2]
        z = z.view(B, T, M, model.d)
        z_tgt = model.encode_target(feats).view(B, T, M, model.d)

        tf = torch.cat([torch.zeros(B, 1, device=dt.device), dt.cumsum(dim=1)], dim=1)
        target = T - 1
        act, t_act, query_t = act_agg[:, :target], tf[:, :target], tf[:, target]
        cuts = sorted({max(1, target // 2), target - 1})

        z_hats, e_norms = [], None
        for k in cuts:
            out = model(z[:, :k + 1], tf[:, :k + 1], act, t_act, query_t, null=False)
            out0 = model(z[:, :k + 1], tf[:, :k + 1], act, t_act, query_t, null=True)
            z_hats.append(out["z_hat"])
            e_norms = (out["z_hat_inv"].float() - out0["z_hat_inv"].float()).norm(dim=-1)
        align, _ = latent_align_loss(z_hats[-1], z_tgt[:, target])
        agree = agreement_loss(z_hats)

        # 反捷径去相关:w vs 下游潜发散(teacher z_inv)/ 像素差
        w = importance_from_effect(e_norms)                       # [B,M]
        fdiv = (z_tgt[:, target, :, d_rev:].float()
                - z_tgt[:, 0, :, d_rev:].float()).norm(dim=-1)    # [B,M]
        pdiff = _patch_pixel_diff(img, 0, target, M)              # [B,M]

        # 闭环 rollout 漂移:用预测作锚点再推一步,看 z_inv 增量幅度
        zr = z_hats[-1]
        ctx = torch.stack([z[:, 0], zr], dim=1)                   # [B,2,M,d]
        tctx = torch.stack([tf[:, 0], query_t], dim=1)
        out2 = model(ctx, tctx, act, t_act, query_t + 1.0, null=False)
        drift = (out2["z_hat_inv"].float() - zr[..., d_rev:].float()).norm(dim=-1).mean()

        tot["align"] += align.item()
        tot["agree"] += agree.item()
        tot["rollout_drift"] += drift.item()
        tot["corr_w_future"] += _pearson(w, fdiv)
        tot["corr_w_pixel"] += _pearson(w, pdiff)
        n += 1

    return {k: v / max(n, 1) for k, v in tot.items()}
