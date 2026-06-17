"""离线诊断评估(序列对齐版):三条独立健康轴 —— 诚实技能比 / 多上下文一致 / 闭环漂移。

主指标 align_ratio = 模型 Δz 误差 / copy-last(复制最近观测帧)误差。<1 才说明模型真在
"预测变化"而非"复制现状";由于锚已改为最近观测帧,agree(多上下文一致)与 rollout_drift
(闭环外推不发散)也都成了非平凡约束。反捷径的 corr(w,future/pixel) 与空间分布留在 viz 面板
(热图比标量曲线更可读,且本身噪声大),不再进 W&B 曲线。
"""
import torch

from train.minecraft._seq import _to_float_img
from train.minecraft.losses import (
    latent_align_loss, agreement_loss, pearson_corr)


def _pearson(a, b):
    """全局 Pearson 相关(float,供监控);与训练 guide 同口径(losses.pearson_corr)。"""
    return float(pearson_corr(a, b))


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


@torch.no_grad()
def evaluate(model, effect_tok, loader, device, amp_dev, use_amp, cfg):
    """holdout 上评估三轴:诚实技能比 align_ratio / 多上下文一致 agree / 闭环漂移 rollout_drift。

    align_ratio = ‖ẑ−z̄_{t*}‖² / ‖copy_last−z̄_{t*}‖²(copy_last=复制最近观测帧 z[:,k]),<1 才算赢。
    loader 为 batch 可迭代对象。反捷径 corr/w 分布见 viz 面板,不在此处计。
    """
    model.eval()
    effect_tok.eval()
    d_rev = model.d_rev
    tot = {"align_ratio": 0.0, "agree": 0.0, "rollout_drift": 0.0}
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
        # autocast 下 adapter 的 Linear 产出 Half;eval 的预测器前向在 autocast 块外
        # (fp32 权重),且下游 z_tgt 统计均为 fp32 —— 统一转 fp32 避免 dtype 不匹配。
        z = z.view(B, T, M, model.d).float()
        z_tgt = model.encode_target(feats).view(B, T, M, model.d).float()

        tf = torch.cat([torch.zeros(B, 1, device=dt.device), dt.cumsum(dim=1)], dim=1)
        target = T - 1
        act, t_act, query_t = act_agg[:, :target], tf[:, :target], tf[:, target]
        cuts = sorted({max(1, target // 2), target - 1})

        z_hats = []
        for k in cuts:
            with torch.autocast(device_type=amp_dev, enabled=use_amp):
                out = model(z[:, :k + 1], tf[:, :k + 1], act, t_act, query_t, null=False)
            z_hats.append(out["z_hat"])
        align, _ = latent_align_loss(z_hats[-1], z_tgt[:, target])
        # 诚实基线:复制最近观测帧 z[:, cuts[-1]](=模型锚帧)对同一未来帧的误差
        copy_last = (z[:, cuts[-1]] - z_tgt[:, target]).pow(2).mean(dim=-1).mean()
        agree = agreement_loss(z_hats)

        # 闭环 rollout 漂移:用预测作锚点再推一步,看 z_inv 增量幅度(锚=最近观测后为真闭环)
        zr = z_hats[-1]
        ctx = torch.stack([z[:, 0], zr], dim=1)                   # [B,2,M,d]
        tctx = torch.stack([tf[:, 0], query_t], dim=1)
        out2 = model(ctx, tctx, act, t_act, query_t + 1.0, null=False)
        drift = (out2["z_hat_inv"].float() - zr[..., d_rev:].float()).norm(dim=-1).mean()

        tot["align_ratio"] += float(align / copy_last.clamp(min=1e-4))
        tot["agree"] += agree.item()
        tot["rollout_drift"] += drift.item()
        n += 1

    return {k: v / max(n, 1) for k, v in tot.items()}
